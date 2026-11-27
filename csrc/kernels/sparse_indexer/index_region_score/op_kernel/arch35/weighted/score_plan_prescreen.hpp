#ifndef IRS_WEIGHTED_SCORE_PLAN_PRESCREEN_HPP
#define IRS_WEIGHTED_SCORE_PLAN_PRESCREEN_HPP

namespace IndexerRegionScoreWeighted {

struct ScoreUniformPrefix {
    uint32_t matchedRequests = 0U;
    uint32_t rowsPerRequest = 0U;
    int32_t firstKv = 0;
};

template <bool CAUSAL>
__aicore__ __attribute__((noinline)) bool EstimateSmallScoreGrid(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, uint32_t qBand,
    ScoreWorkCosts costs, uint64_t planningCost, ScoreUniformPrefix prefix = {})
{
    const auto bounds = reinterpret_cast<__gm__ const int32_t *>(cu);
    const auto lengths = reinterpret_cast<__gm__ const int32_t *>(kv);
    const DeviceScoreEpoch all{0U, tiling.totalTokens, 0U, tiling.numRequests};
    uint32_t shard = 0U;
    uint64_t gridPeak = 0U;
    uint64_t gridLoad = 0U;
    uint64_t total = 0U;
    int32_t previous;
    if (prefix.matchedRequests != 0U) {
        // The entry has verified complete identical requests aligned with the
        // original Q grid. Their work and completed shard peaks repeat exactly.
        const auto shape = MakeScoreWorkShape<CAUSAL>(tiling, all, 0,
            static_cast<int32_t>(prefix.rowsPerRequest), prefix.firstKv, qBand);
        const uint64_t requestWork = ScoreWork(shape, costs);
        const uint32_t requestsPerShard = tiling.numRequests / tiling.qShardCount;
        shard = prefix.matchedRequests / requestsPerShard;
        const uint32_t remaining = prefix.matchedRequests % requestsPerShard;
        total = requestWork * prefix.matchedRequests;
        gridLoad = requestWork * remaining;
        if (shard != 0U) {
            gridPeak = ShareScoreGridWork(requestWork * requestsPerShard, tiling.kShardCount);
        }
        previous = static_cast<int32_t>(prefix.matchedRequests * prefix.rowsPerRequest);
    } else {
        previous = bounds[0];
    }
    uint32_t shardEnd = ScoreGridQEnd(tiling, all, shard + 1U);
    for (uint32_t request = prefix.matchedRequests; request < tiling.numRequests; ++request) {
        const int32_t next = bounds[request + 1U];
        if (previous < 0 || next < previous || static_cast<uint32_t>(next) > tiling.totalTokens) {
            return true;
        }
        const uint32_t start = static_cast<uint32_t>(previous);
        const uint32_t stop = static_cast<uint32_t>(next);
        previous = next;
        if (start == stop) {
            continue;
        }
        const int32_t kvLength = CAUSAL ? lengths[request] : 0;
        const auto shape = MakeScoreWorkShape<CAUSAL>(tiling, all, static_cast<int32_t>(start), next,
            kvLength, qBand);
        const uint64_t requestWork = ScoreWork(shape, costs);
        total += requestWork;
        uint32_t begin = start;
        while (begin < stop) {
            const uint32_t end = PagedMin(shardEnd, stop);
            if (end > begin) {
                const uint64_t work = begin == start && end == stop ? requestWork :
                    ScoreWork(ClipScoreWork(shape, begin, end), costs);
                gridLoad += work;
                begin = end;
            }
            if (begin < shardEnd || shard + 1U >= tiling.qShardCount) {
                break;
            }
            const uint64_t shardWork = ShareScoreGridWork(gridLoad, tiling.kShardCount);
            gridPeak = shardWork > gridPeak ? shardWork : gridPeak;
            gridLoad = 0U;
            shardEnd = ScoreGridQEnd(tiling, all, ++shard + 1U);
        }
    }
    const uint64_t shardWork = ShareScoreGridWork(gridLoad, tiling.kShardCount);
    gridPeak = shardWork > gridPeak ? shardWork : gridPeak;
    return PreferScoreCuts(total, gridPeak, tiling.usedCubeCores, costs, planningCost);
}

template <bool CAUSAL>
__aicore__ inline bool ShouldPrepareScorePartition(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, uint32_t qBand,
    ScoreUniformPrefix prefix = {})
{
    const uint32_t pairs = tiling.usedCubeCores;
    if (pairs <= 1U) {
        return false;
    }
    const uint64_t qTiles = (static_cast<uint64_t>(tiling.totalTokens) + tiling.qTile - 1U) / tiling.qTile;
    const uint64_t maxQ = (qTiles + tiling.qShardCount - 1U) / tiling.qShardCount * tiling.qTile;
    const uint64_t kMacros = (tiling.seqK + PAGED_K_MACRO - 1U) / PAGED_K_MACRO;
    const uint64_t maxN = ((kMacros + tiling.kShardCount - 1U) / tiling.kShardCount) *
        (PAGED_K_MACRO / PAGED_NR);
    if (tiling.numRequests == 1U && tiling.kShardCount == 1U && kMacros <= 1U && maxQ <= qBand) {
        // Each original pair owns at most one Q band and one K macro. Its
        // entire cost fits the cut-rounding/reload allowance in PreferScoreCuts,
        // so that predicate cannot select weighted ownership for any KV length.
        // The same bound covers every smaller bounded-score epoch.
        return false;
    }
    const ScoreWorkCosts costs = GetScoreWorkCosts(tiling, qBand);
    const uint64_t publicationCost = 4U * pairs;
    // ScoreRequestTable::Init scans four uint32 prefix counters per pair
    // before cuts can be selected. Charge that scan before starting planning.
    const uint64_t prefixScanCost = 4U * pairs;
    const uint64_t planningCost = publicationCost + prefixScanCost + (tiling.numRequests + 1U) / 2U;
    if (tiling.numRequests == 1U) {
        // An exactly tiled rectangle already attains the grid lower bound.
        if constexpr (!CAUSAL) {
            if (tiling.totalTokens % qBand == 0U && qTiles % tiling.qShardCount == 0U &&
                tiling.seqK % PAGED_K_MACRO == 0U && kMacros % tiling.kShardCount == 0U) {
                return false;
            }
        }
        return true;
    }
    const uint64_t maxFragments = maxQ < tiling.numRequests ? maxQ : tiling.numRequests;
    if (maxN * (costs.key * maxFragments + costs.tile * maxQ) <= planningCost) {
        // Every nonempty request fragment owns at least one Q row. Charge
        // its Key load and bound every row by a complete Q band so this
        // shortcut remains valid for arbitrary request boundaries.
        return false;
    }
    // Small metadata scans are cheaper than publishing a full solve. Larger
    // request lists stay with the shared planner and are never rescanned by
    // every pair.
    if (tiling.numRequests <= SCORE_METADATA_CHUNK / 4U && cu != nullptr) {
        return EstimateSmallScoreGrid<CAUSAL>(tiling, cu, kv, qBand, costs, planningCost, prefix);
    }
    return true;
}

} // namespace IndexerRegionScoreWeighted

#endif
