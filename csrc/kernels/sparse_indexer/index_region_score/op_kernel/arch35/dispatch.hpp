#ifndef IRS_DEVICE_DISPATCH_HPP
#define IRS_DEVICE_DISPATCH_HPP
#include <type_traits>

namespace IndexerRegionScorePaged {

struct DispatchShape {
    uint32_t numRequests = 0U;
    uint32_t totalTokens = 0U;
    uint32_t qShardCount = 0U;
    uint32_t qTile = 0U;
    uint32_t batchTokens = 0U;
};

// Every participant reads the same immutable metadata and reaches the same branch.
// Only equal complete requests with matching grid boundaries are certified here.
template <bool CAUSAL>
__aicore__ inline bool IsUniformRequestGrid(const DispatchShape &t, GM_ADDR cu, GM_ADDR kv,
    IndexerRegionScoreWeighted::ScoreUniformPrefix &prefix)
{
    if (t.numRequests == 0U || cu == nullptr || t.qShardCount == 0U || t.qTile == 0U ||
        t.totalTokens % t.numRequests != 0U || t.numRequests % t.qShardCount != 0U) {
        return false;
    }
    if (t.batchTokens != 0U && t.batchTokens < t.totalTokens) {
        return false;
    }
    const uint32_t rows = t.totalTokens / t.numRequests;
    if (rows == 0U || rows % t.qTile != 0U) {
        return false;
    }
    // Bound the replicated scan by one metadata window. Larger request lists
    // use the shared planner's parallel metadata loading.
    constexpr uint32_t metadataRequestWords = 2U; // One Q boundary and one KV length.
    constexpr uint32_t maxReplicatedRequests =
        IndexerRegionScoreWeighted::SCORE_METADATA_CHUNK / metadataRequestWords;
    if (t.numRequests > maxReplicatedRequests) {
        return false;
    }
    const auto bounds = reinterpret_cast<__gm__ const int32_t *>(cu);
    const auto lengths = reinterpret_cast<__gm__ const int32_t *>(kv);
    if (bounds[0] != 0) {
        return false;
    }
    int32_t firstKv = 0;
    if constexpr (CAUSAL) {
        if (kv == nullptr) {
            return false;
        }
        firstKv = lengths[0];
    }
    uint32_t expected = 0U;
    uint32_t request = 0U;
    for (; request < t.numRequests; ++request) {
        expected += rows;
        if (bounds[request + 1U] != static_cast<int32_t>(expected)) {
            break;
        }
        if constexpr (CAUSAL) {
            if (lengths[request] != firstKv) {
                break;
            }
        }
    }
    if (request == t.numRequests) {
        return true;
    }
    prefix = {request, rows, firstKv};
    return false;
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ inline void RunWithStrategies(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
    GM_ADDR actualKvLengths, GM_ADDR pagedKeySummary, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    DispatchShape decoded{};
    PagedTilingData schedule;
    if constexpr (CacheStrategy::IS_PAGED) {
        const auto words = reinterpret_cast<__gm__ const uint32_t *>(tiling);
        decoded.numRequests = words[offsetof(PagedTilingData, numRequests) / sizeof(uint32_t)];
        decoded.totalTokens = words[offsetof(PagedTilingData, totalTokens) / sizeof(uint32_t)];
        decoded.qShardCount = words[offsetof(PagedTilingData, qShardCount) / sizeof(uint32_t)];
        decoded.qTile = words[offsetof(PagedTilingData, qTile) / sizeof(uint32_t)];
        decoded.batchTokens = words[offsetof(PagedTilingData, batchTokens) / sizeof(uint32_t)];
    } else {
        // Continuous qTile is derived by normalization, rather than its POD qTile field.
        GET_TILING_DATA_WITH_STRUCT(RegionScoreTilingData, data, tiling);
        schedule = NormalizeContinuousTiling(data);
        decoded = {schedule.numRequests, schedule.totalTokens, schedule.qShardCount,
            schedule.qTile, schedule.batchTokens};
    }
    IndexerRegionScoreWeighted::ScoreUniformPrefix prefix{};
    bool direct = IsUniformRequestGrid<MaskStrategy::CAUSAL>(decoded, cuSeqLens, actualKvLengths, prefix);
    if (!direct) {
        if constexpr (CacheStrategy::IS_PAGED) {
            const auto words = reinterpret_cast<__gm__ const uint32_t *>(tiling);
            schedule.numRequests = decoded.numRequests;
            schedule.totalTokens = decoded.totalTokens;
            schedule.qShardCount = decoded.qShardCount;
            schedule.qTile = decoded.qTile;
            schedule.batchTokens = decoded.batchTokens;
            schedule.kShardCount = words[offsetof(PagedTilingData, kShardCount) / sizeof(uint32_t)];
            schedule.usedCubeCores = words[offsetof(PagedTilingData, usedCubeCores) / sizeof(uint32_t)];
            schedule.seqK = words[offsetof(PagedTilingData, seqK) / sizeof(uint32_t)];
            schedule.regionSize = words[offsetof(PagedTilingData, regionSize) / sizeof(uint32_t)];
            schedule.groups = words[offsetof(PagedTilingData, groups) / sizeof(uint32_t)];
            schedule.headsPerGroup = words[offsetof(PagedTilingData, headsPerGroup) / sizeof(uint32_t)];
            schedule.pageCols = words[offsetof(PagedTilingData, pageCols) / sizeof(uint32_t)];
            schedule.summariesPerPage = words[offsetof(PagedTilingData, summariesPerPage) / sizeof(uint32_t)];
        }
        GM_ADDR planCu = (blockTable == nullptr && schedule.numRequests == 1U) ? nullptr : cuSeqLens;
        direct = !IndexerRegionScoreWeighted::ShouldPrepareScorePartition<MaskStrategy::CAUSAL>(
            schedule, planCu, actualKvLengths, IndexerRegionScoreWeighted::ScoreQueryBand(schedule), prefix);
    }
    if (direct) {
        RunGridWithStrategies<CacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
        return;
    }
    using WeightedMask = typename std::conditional<MaskStrategy::CAUSAL,
        IndexerRegionScoreWeighted::RightDownCausalMaskStrategy,
        IndexerRegionScoreWeighted::NoMaskStrategy>::type;
    if ASCEND_IS_AIC {
        IndexerRegionScoreWeighted::RunCubeWithStrategies<CacheStrategy, WeightedMask, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
    } else {
        IndexerRegionScoreWeighted::RunVectorWithStrategies<CacheStrategy, WeightedMask, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
    }
}
} // namespace IndexerRegionScorePaged
#endif
