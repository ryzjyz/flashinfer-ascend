#ifndef IRS_WEIGHTED_DEVICE_SCORE_PLAN_HPP
#define IRS_WEIGHTED_DEVICE_SCORE_PLAN_HPP

#include "device_work_math.hpp"

namespace IndexerRegionScoreWeighted {
constexpr uint32_t SCORE_PLAN_LINE_WORDS = IndexerRegionScore::SCORE_PLAN_LINE_BYTES / sizeof(uint32_t);

__aicore__ inline uint64_t DeviceScorePlanBytes(uint32_t requests, uint32_t pairs,
    uint32_t totalTokens, uint32_t batchTokens)
{
    return IndexerRegionScore::ScorePlanWorkspaceBytes(requests, pairs, totalTokens, batchTokens);
}

__aicore__ inline uint64_t ScoreCutSlotBytes(uint32_t pairs)
{
    return (static_cast<uint64_t>(pairs) + 2U) * IndexerRegionScore::SCORE_PLAN_LINE_BYTES;
}

struct DeviceScoreCut {
    uint32_t request = 0U;
    uint32_t kMacro = 0U;
    uint32_t qBand = 0U;
};

struct DeviceScoreEpoch {
    uint32_t qBegin;
    uint32_t qEnd;
    uint32_t rBegin;
    uint32_t rEnd;
};

__aicore__ inline uint32_t LocateScoreRequest(GM_ADDR cu, uint32_t requests, uint32_t token)
{
    if (requests <= 1U || cu == nullptr) {
        return 0U;
    }
    auto bounds = reinterpret_cast<__gm__ int32_t *>(cu);
    uint32_t left = 0U;
    uint32_t right = requests;
    while (left < right) {
        const uint32_t mid = left + (right - left) / 2U;
        if (bounds[mid + 1U] <= static_cast<int32_t>(token)) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    return left;
}

__aicore__ inline DeviceScoreEpoch GetDeviceScoreEpoch(const PagedTilingData &tiling, GM_ADDR cu)
{
    const uint32_t qBegin = tiling.batchTokens == 0U ? 0U : tiling.batchBegin;
    const uint32_t qEnd = tiling.batchTokens == 0U
        ? tiling.totalTokens : tiling.batchBegin + tiling.batchCount;
    if (qBegin >= qEnd || tiling.numRequests == 0U) {
        return {qBegin, qEnd, 0U, 0U};
    }
    return {qBegin, qEnd, LocateScoreRequest(cu, tiling.numRequests, qBegin),
            PagedMin(LocateScoreRequest(cu, tiling.numRequests, qEnd - 1U) + 1U, tiling.numRequests)};
}

template <bool CAUSAL>
__aicore__ inline DeviceWorkShape MakeScoreWorkShape(
    const PagedTilingData &tiling, DeviceScoreEpoch epoch, int32_t low, int32_t high,
    int32_t kvLength, uint32_t qBand)
{
    if (low < 0 || high <= low || static_cast<uint32_t>(high) > tiling.totalTokens) {
        return {0U, 0U, qBand, PAGED_K_MACRO, tiling.seqK, tiling.regionSize, 0, CAUSAL};
    }
    const uint32_t begin = static_cast<uint32_t>(low);
    const uint32_t end = static_cast<uint32_t>(high);
    const uint32_t qBegin = PagedMax(begin, epoch.qBegin);
    const uint32_t qEnd = PagedMin(end, epoch.qEnd);
    int64_t position = 0;
    if constexpr (CAUSAL) {
        if (qBegin < qEnd) {
            position = RequestFirstTokenPosition(kvLength, begin, end) + (qBegin - begin);
        }
    }
    return {qBegin, PagedMax(qBegin, qEnd), qBand, PAGED_K_MACRO, tiling.seqK,
            tiling.regionSize, position, CAUSAL, tiling.headsPerGroup};
}

template <bool CAUSAL>
__aicore__ inline DeviceWorkShape GetDeviceWorkShape(
    const PagedTilingData &tiling, DeviceScoreEpoch epoch, GM_ADDR cu, GM_ADDR kv,
    uint32_t request, uint32_t qBand)
{
    int32_t begin = 0;
    int32_t end = static_cast<int32_t>(tiling.totalTokens);
    if (cu != nullptr) {
        const auto bounds = reinterpret_cast<__gm__ int32_t *>(cu);
        begin = bounds[request];
        end = bounds[request + 1U];
    }
    int32_t kvLength = 0;
    if constexpr (CAUSAL) {
        kvLength = reinterpret_cast<__gm__ int32_t *>(kv)[request];
    }
    return MakeScoreWorkShape<CAUSAL>(tiling, epoch, begin, end, kvLength, qBand);
}

constexpr uint32_t SCORE_METADATA_CHUNK = 128U;
constexpr uint32_t SCORE_KV_BUFFER_OFFSET = SCORE_METADATA_CHUNK + 8U;

template <bool CAUSAL>
__aicore__ inline void LoadScoreMetadata(LocalTensor<int32_t> buffer, GM_ADDR cu, GM_ADDR kv,
    uint32_t request, uint32_t count)
{
    GlobalTensor<int32_t> boundaries;
    boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cu));
    const DataCopyPadExtParams<int32_t> padding{false, 0U, 0U, 0};
    DataCopyPad(buffer, boundaries[request], {1U, (count + 1U) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, padding);
    if constexpr (CAUSAL) {
        GlobalTensor<int32_t> lengths;
        lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(kv));
        DataCopyPad(buffer[SCORE_KV_BUFFER_OFFSET], lengths[request],
                    {1U, count * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, padding);
    }
    const auto event = GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>();
    SetFlag<HardEvent::MTE2_S>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
}

template <bool CAUSAL>
__aicore__ inline DeviceWorkShape GetLoadedScoreWork(const PagedTilingData &tiling,
    const DeviceScoreEpoch &epoch, LocalTensor<int32_t> buffer, uint32_t index, uint32_t qBand)
{
    const int32_t length = CAUSAL ? buffer.GetValue(SCORE_KV_BUFFER_OFFSET + index) : 0;
    return MakeScoreWorkShape<CAUSAL>(
        tiling, epoch, buffer.GetValue(index), buffer.GetValue(index + 1U), length, qBand);
}

__aicore__ inline void FlushScorePlanLine(GlobalTensor<uint32_t> &gm, uint64_t offset)
{
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm[offset]);
}

__aicore__ inline void WriteScoreCut(
    GlobalTensor<uint32_t> &gm, uint32_t id, const DeviceScoreCut &cut)
{
    const uint32_t offset = (id) * SCORE_PLAN_LINE_WORDS;
    gm.SetValue(offset, cut.request);
    gm.SetValue(offset + 1U, cut.kMacro);
    gm.SetValue(offset + 2U, cut.qBand);
    FlushScorePlanLine(gm, offset);
}

__aicore__ inline DeviceScoreCut ReadScoreCut(GM_ADDR plan, uint32_t id)
{
    GlobalTensor<uint32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(plan));
    const uint32_t offset = (id) * SCORE_PLAN_LINE_WORDS;
    FlushScorePlanLine(gm, offset);
    return {gm.GetValue(offset), gm.GetValue(offset + 1U), gm.GetValue(offset + 2U)};
}

__aicore__ inline uint64_t ScoreTarget(uint64_t total, uint32_t pairs, uint32_t id)
{
    return total / pairs * id + (total % pairs) * id / pairs;
}

struct ScorePartition {
    DeviceScoreCut begin{};
    DeviceScoreCut end{};
    bool weighted = false;
};

struct ScoreWorkCosts {
    uint32_t key;
    uint32_t tile;
};

__aicore__ inline ScoreWorkCosts GetScoreWorkCosts(const PagedTilingData &tiling, uint32_t qBand)
{
    const uint32_t keyLoadCost = tiling.pageCols == 0U ? 1U :
        SparseIndexer::Common::CeilDiv(PAGED_NR, PagedMax(1U, tiling.summariesPerPage));
    return {keyLoadCost,
        PagedMax(1U, tiling.groups * ((qBand * tiling.headsPerGroup + 15U) / 16U))};
}

__aicore__ inline uint32_t ScoreGridQEnd(const PagedTilingData &tiling,
    const DeviceScoreEpoch &epoch, uint32_t shardEnd)
{
    const uint32_t qCount = epoch.qEnd - epoch.qBegin;
    const uint32_t qTiles = SparseIndexer::Common::CeilDiv(qCount, tiling.qTile);
    const uint32_t shards = tiling.qShardCount;
    const uint64_t end = static_cast<uint64_t>(shardEnd * (qTiles / shards) +
        PagedMin(shardEnd, qTiles % shards)) * tiling.qTile;
    return epoch.qBegin + (end < qCount ? static_cast<uint32_t>(end) : qCount);
}

__aicore__ inline uint64_t ScoreWork(const DeviceWorkShape &shape, ScoreWorkCosts costs)
{
    const uint32_t macros = KMacroCount(shape);
    if (macros != 0U && BandCount(shape) == 1U) {
        const uint32_t microtiles = (macros - 1U) * (PAGED_K_MACRO / PAGED_NR) +
            KeyMicrotiles(shape, macros - 1U);
        return static_cast<uint64_t>(microtiles) * (costs.key + TailBandCost(shape, costs.tile));
    }
    return WorkPrefix(shape, macros, costs.key, costs.tile);
}

__aicore__ inline DeviceWorkShape ClipScoreWork(DeviceWorkShape shape, uint32_t begin, uint32_t end)
{
    shape.firstPosition += begin - shape.qBegin;
    shape.qBegin = begin;
    shape.qEnd = end;
    return shape;
}

__aicore__ inline uint64_t ShareScoreGridWork(uint64_t cost, uint32_t kShards)
{
    return cost / kShards + (cost % kShards != 0U);
}

__aicore__ inline uint64_t ScoreCutBoundaryCost(ScoreWorkCosts costs)
{
    // A target rounds forward by at most one complete Q band. The following
    // pair may reload the same K macro. Bound both by a full-width macro.
    return (static_cast<uint64_t>(costs.tile) + costs.key) * (PAGED_K_MACRO / PAGED_NR);
}

__aicore__ inline bool PreferScoreCuts(uint64_t total, uint64_t gridPeak, uint32_t pairs,
    ScoreWorkCosts costs, uint64_t planningCost = 0U)
{
    // Every participant applies the same global bound before computing cuts.
    return gridPeak > ShareScoreGridWork(total, pairs) + ScoreCutBoundaryCost(costs) + planningCost;
}

__aicore__ inline DeviceScoreCut FindScoreCut(const DeviceWorkShape &shape,
    uint32_t request, uint64_t target, ScoreWorkCosts costs)
{
    DeviceScoreCut result{request, 0U, 0U};
    SeekCut(shape, target, costs.key, costs.tile, result.kMacro, result.qBand);
    return result;
}

template <bool CAUSAL>
__aicore__ __attribute__((noinline)) ScorePartition MakeSingleRequestPartition(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, uint32_t qBand,
    uint32_t pair, ScoreWorkCosts costs)
{
    const auto epoch = GetDeviceScoreEpoch(tiling, cu);
    const auto shape = GetDeviceWorkShape<CAUSAL>(tiling, epoch, cu, kv, 0U, qBand);
    if (shape.qBegin == shape.qEnd || tiling.usedCubeCores <= 1U) {
        return {};
    }
    const uint64_t total = ScoreWork(shape, costs);
    if (total == 0U) {
        return {};
    }
    const uint32_t qShards = tiling.qShardCount;
    const uint32_t qTiles = SparseIndexer::Common::CeilDiv(epoch.qEnd - epoch.qBegin, tiling.qTile);
    const uint32_t lastWideShard = qTiles % qShards;
    const uint32_t tailBegin = PagedMax(ScoreGridQEnd(tiling, epoch, qShards - 1U), shape.qBegin);
    uint64_t peak = tailBegin < shape.qEnd
        ? ScoreWork(ClipScoreWork(shape, tailBegin, shape.qEnd), costs) : 0U;
    if (lastWideShard != 0U) {
        const uint32_t begin = PagedMax(ScoreGridQEnd(tiling, epoch, lastWideShard - 1U), shape.qBegin);
        const uint32_t end = PagedMin(ScoreGridQEnd(tiling, epoch, lastWideShard), shape.qEnd);
        const uint64_t wide = begin < end ? ScoreWork(ClipScoreWork(shape, begin, end), costs) : 0U;
        peak = peak > wide ? peak : wide;
    }
    if (!PreferScoreCuts(total, ShareScoreGridWork(peak, tiling.kShardCount),
                         tiling.usedCubeCores, costs)) {
        return {};
    }
    return {FindScoreCut(shape, 0U, ScoreTarget(total, tiling.usedCubeCores, pair), costs),
            FindScoreCut(shape, 0U, ScoreTarget(total, tiling.usedCubeCores, pair + 1U), costs), true};
}

constexpr uint32_t SCORE_REQUEST_WORDS = 8U;
constexpr uint32_t SCORE_REQUESTS_PER_LINE = SCORE_PLAN_LINE_WORDS / SCORE_REQUEST_WORDS;

struct ScorePrefix {
    uint64_t work = 0U;
    uint64_t handoff = 0U;
};

__aicore__ inline ScorePrefix operator+(ScorePrefix left, ScorePrefix right)
{
    return {left.work + right.work, left.handoff + right.handoff};
}

__aicore__ inline ScorePrefix operator-(ScorePrefix left, ScorePrefix right)
{
    return {left.work - right.work, left.handoff - right.handoff};
}

__aicore__ inline void WriteScoreCounter(GlobalTensor<uint32_t> &gm, uint64_t offset, uint64_t value)
{
    gm.SetValue(offset, static_cast<uint32_t>(value));
    gm.SetValue(offset + 1U, static_cast<uint32_t>(value >> 32U));
}

__aicore__ inline uint64_t ReadScoreCounter(GlobalTensor<uint32_t> &gm, uint64_t offset)
{
    return gm.GetValue(offset) | (static_cast<uint64_t>(gm.GetValue(offset + 1U)) << 32U);
}

__aicore__ inline void WriteScorePrefix(GlobalTensor<uint32_t> &gm, uint64_t offset, ScorePrefix prefix)
{
    WriteScoreCounter(gm, offset, prefix.work);
    WriteScoreCounter(gm, offset + 2U, prefix.handoff);
}

__aicore__ inline ScorePrefix ReadScorePrefix(GlobalTensor<uint32_t> &gm, uint64_t offset)
{
    return {ReadScoreCounter(gm, offset), ReadScoreCounter(gm, offset + 2U)};
}

__aicore__ inline uint64_t WorkAtPublishedCut(
    const DeviceWorkShape &shape, const DeviceScoreCut &cut, ScoreWorkCosts costs, uint64_t &reload)
{
    const uint32_t kMacro = cut.kMacro;
    reload = 0U;
    uint64_t prefix = WorkPrefix(shape, kMacro, costs.key, costs.tile);
    if (kMacro < KMacroCount(shape)) {
        const uint32_t firstBand = FirstActiveBand(shape, kMacro);
        if (cut.qBand > firstBand) {
            const uint32_t width = KeyMicrotiles(shape, kMacro);
            reload = static_cast<uint64_t>(costs.key) * width;
            prefix += reload + static_cast<uint64_t>(cut.qBand - firstBand) * costs.tile * width;
        }
    }
    return prefix;
}

__aicore__ inline uint64_t ExtraScheduleWork(const DeviceWorkShape &shape, uint32_t kMacro)
{
    const uint32_t firstBandEnd = BandKEnd(shape, 0U);
    const uint32_t distinctStarts = kMacro > firstBandEnd ? kMacro - firstBandEnd : 0U;
    return static_cast<uint64_t>(distinctStarts) * (PAGED_K_MACRO / PAGED_NR);
}

__aicore__ inline uint32_t ScoreRequestsPerPair(uint32_t requests, uint32_t pairs)
{
    using SparseIndexer::Common::CeilDiv;
    return CeilDiv(CeilDiv(requests, SCORE_REQUESTS_PER_LINE), pairs) * SCORE_REQUESTS_PER_LINE;
}

template <bool CAUSAL>
__aicore__ __attribute__((noinline)) void PrepareScoreRequests(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, GM_ADDR descriptors, GM_ADDR summaries,
    uint32_t qBand, uint32_t pair, ScoreWorkCosts costs, LocalTensor<int32_t> metadata)
{
    GlobalTensor<uint32_t> requests;
    requests.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(descriptors));
    const uint32_t width = ScoreRequestsPerPair(tiling.numRequests, tiling.usedCubeCores);
    const uint32_t begin = pair * width;
    const uint32_t end = PagedMin(begin + width, tiling.numRequests);
    const DeviceScoreEpoch all{0U, tiling.totalTokens, 0U, tiling.numRequests};
    ScorePrefix prefix;
    for (uint32_t start = begin; start < end; start += SCORE_METADATA_CHUNK) {
        const uint32_t count = PagedMin(end - start, SCORE_METADATA_CHUNK);
        LoadScoreMetadata<CAUSAL>(metadata, cu, kv, start, count);
        for (uint32_t index = 0U; index < count; ++index) {
            const auto shape = GetLoadedScoreWork<CAUSAL>(tiling, all, metadata, index, qBand);
            const uint64_t offset = static_cast<uint64_t>(start + index) * SCORE_REQUEST_WORDS;
            requests.SetValue(offset, shape.qBegin);
            requests.SetValue(offset + 1U, shape.qEnd);
            WriteScoreCounter(requests, offset + 2U, static_cast<uint64_t>(shape.firstPosition));
            WriteScorePrefix(requests, offset + 4U, prefix);
            prefix = prefix + ScorePrefix{ScoreWork(shape, costs), ExtraScheduleWork(shape, KMacroCount(shape))};
            if ((index + 1U) % SCORE_REQUESTS_PER_LINE == 0U || index + 1U == count) {
                FlushScorePlanLine(requests, offset - offset % SCORE_PLAN_LINE_WORDS);
            }
        }
    }
    GlobalTensor<uint32_t> totals;
    totals.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(summaries));
    const uint32_t offset = pair * SCORE_PLAN_LINE_WORDS;
    WriteScorePrefix(totals, offset, prefix);
    FlushScorePlanLine(totals, offset);
}

class ScoreRequestTable {
public:
    __aicore__ inline void Init(const PagedTilingData &tiling, GM_ADDR descriptors, GM_ADDR summaries,
        LocalTensor<int32_t> metadata)
    {
        requests_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(descriptors));
        blocks_ = metadata.template ReinterpretCast<uint32_t>();
        count_ = tiling.numRequests;
        width_ = ScoreRequestsPerPair(count_, tiling.usedCubeCores);
        GlobalTensor<uint32_t> totals;
        totals.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(summaries));
        constexpr uint32_t bytes = 32U;
        DataCopyPad(blocks_, totals, {static_cast<uint16_t>(tiling.usedCubeCores), bytes,
            IndexerRegionScore::SCORE_PLAN_LINE_BYTES - bytes, 0U, 0U}, {false, 0U, 0U, 0U});
        const auto event = GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        for (uint32_t pair = 0U; pair < tiling.usedCubeCores; ++pair) {
            const uint32_t offset = pair * bytes / sizeof(uint32_t);
            const ScorePrefix next{LocalCounter(offset), LocalCounter(offset + 2U)};
            blocks_.SetValue(offset, static_cast<uint32_t>(total_.work));
            blocks_.SetValue(offset + 1U, static_cast<uint32_t>(total_.work >> 32U));
            blocks_.SetValue(offset + 2U, static_cast<uint32_t>(total_.handoff));
            blocks_.SetValue(offset + 3U, static_cast<uint32_t>(total_.handoff >> 32U));
            total_ = total_ + next;
        }
    }

    __aicore__ inline ScorePrefix Prefix(uint32_t request)
    {
        if (request == count_) {
            return total_;
        }
        const uint32_t block = (request / width_) * 8U;
        const uint64_t offset = RequestOffset(request);
        return ScorePrefix{LocalCounter(block), LocalCounter(block + 2U)} +
            ReadScorePrefix(requests_, offset + 4U);
    }

    template <bool CAUSAL>
    __aicore__ inline DeviceWorkShape Shape(const PagedTilingData &tiling, uint32_t request, uint32_t qBand)
    {
        const uint64_t offset = RequestOffset(request);
        return {requests_.GetValue(offset), requests_.GetValue(offset + 1U), qBand,
            PAGED_K_MACRO, tiling.seqK, tiling.regionSize,
            static_cast<int64_t>(ReadScoreCounter(requests_, offset + 2U)), CAUSAL, tiling.headsPerGroup};
    }

private:
    __aicore__ inline uint64_t RequestOffset(uint32_t request)
    {
        const uint64_t offset = static_cast<uint64_t>(request) * SCORE_REQUEST_WORDS;
        FlushScorePlanLine(requests_, offset - offset % SCORE_PLAN_LINE_WORDS);
        return offset;
    }

    __aicore__ inline uint64_t LocalCounter(uint32_t offset)
    {
        return blocks_.GetValue(offset) | (static_cast<uint64_t>(blocks_.GetValue(offset + 1U)) << 32U);
    }

    GlobalTensor<uint32_t> requests_;
    LocalTensor<uint32_t> blocks_;
    uint32_t count_;
    uint32_t width_;
    ScorePrefix total_{};
};

template <bool CAUSAL>
class ScoreEpochTable {
public:
    __aicore__ inline ScoreEpochTable(const PagedTilingData &tiling, DeviceScoreEpoch epoch,
        uint32_t qBand, ScoreWorkCosts costs, ScoreRequestTable &requests)
        : tiling_(tiling), epoch_(epoch), qBand_(qBand), costs_(costs), requests_(requests)
    {
        if (epoch.rBegin >= epoch.rEnd) {
            return;
        }
        const auto first = Shape(epoch.rBegin);
        firstWork_ = Work(first);
        total_ = firstWork_;
        if (epoch.rBegin + 1U < epoch.rEnd) {
            wholeStart_ = requests_.Prefix(epoch.rBegin + 1U);
            const uint32_t last = epoch.rEnd - 1U;
            total_ = total_ + (requests_.Prefix(last) - wholeStart_) + Work(Shape(last));
        }
    }

    __aicore__ inline ScorePrefix Total() const { return total_; }

    __aicore__ inline ScorePrefix Prefix(uint32_t request)
    {
        if (request <= epoch_.rBegin) {
            return {};
        }
        if (request >= epoch_.rEnd) {
            return total_;
        }
        return firstWork_ + (requests_.Prefix(request) - wholeStart_);
    }

    __aicore__ inline DeviceScoreCut Cut(uint64_t target, uint64_t &position, uint64_t &reload)
    {
        reload = 0U;
        if (target >= total_.work) {
            position = total_.work + total_.handoff;
            return {epoch_.rEnd, 0U, 0U};
        }
        uint32_t lo = epoch_.rBegin;
        uint32_t hi = epoch_.rEnd;
        while (lo + 1U < hi) {
            const uint32_t mid = lo + (hi - lo) / 2U;
            if (Prefix(mid).work <= target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const auto prefix = Prefix(lo);
        const auto shape = Shape(lo);
        const auto boundary = FindScoreCut(shape, lo, target - prefix.work, costs_);
        position = prefix.work + prefix.handoff + WorkAtPublishedCut(shape, boundary, costs_, reload) +
            ExtraScheduleWork(shape, boundary.kMacro);
        return boundary;
    }

private:
    __aicore__ inline ScorePrefix Work(const DeviceWorkShape &shape)
    {
        return {ScoreWork(shape, costs_), ExtraScheduleWork(shape, KMacroCount(shape))};
    }

    __aicore__ inline DeviceWorkShape Shape(uint32_t request)
    {
        auto shape = requests_.template Shape<CAUSAL>(tiling_, request, qBand_);
        const uint32_t begin = PagedMax(shape.qBegin, epoch_.qBegin);
        const uint32_t end = PagedMin(shape.qEnd, epoch_.qEnd);
        return ClipScoreWork(shape, begin, PagedMax(begin, end));
    }

    const PagedTilingData &tiling_;
    DeviceScoreEpoch epoch_;
    uint32_t qBand_;
    ScoreWorkCosts costs_;
    ScoreRequestTable &requests_;
    ScorePrefix firstWork_{};
    ScorePrefix wholeStart_{};
    ScorePrefix total_{};
};

__aicore__ inline DeviceScoreEpoch ScoreEpochForSpan(const PagedTilingData &tiling, GM_ADDR cu,
    uint32_t begin, uint32_t end)
{
    if (begin >= end) {
        return {begin, end, 0U, 0U};
    }
    return {begin, end, LocateScoreRequest(cu, tiling.numRequests, begin),
        PagedMin(LocateScoreRequest(cu, tiling.numRequests, end - 1U) + 1U, tiling.numRequests)};
}

template <bool CAUSAL>
__aicore__ __attribute__((noinline)) void WriteScorePairPlan(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR slot, uint32_t qBand, ScoreWorkCosts costs,
    ScoreRequestTable &requests, uint32_t pair)
{
    const auto epoch = GetDeviceScoreEpoch(tiling, cu);
    ScoreEpochTable<CAUSAL> work(tiling, epoch, qBand, costs, requests);
    uint64_t startPosition;
    uint64_t endPosition;
    uint64_t reload;
    uint64_t unused;
    const auto begin = work.Cut(ScoreTarget(work.Total().work, tiling.usedCubeCores, pair),
        startPosition, reload);
    const auto end = work.Cut(ScoreTarget(work.Total().work, tiling.usedCubeCores, pair + 1U),
        endPosition, unused);
    const uint64_t candidate = endPosition > startPosition ? endPosition - startPosition + reload : 0U;
    uint64_t direct = 0U;
    if (pair < tiling.qShardCount) {
        const auto part = ScoreEpochForSpan(tiling, cu, ScoreGridQEnd(tiling, epoch, pair),
            ScoreGridQEnd(tiling, epoch, pair + 1U));
        ScoreEpochTable<CAUSAL> grid(tiling, part, qBand, costs, requests);
        direct = ShareScoreGridWork(grid.Total().work, tiling.kShardCount);
    }
    GlobalTensor<uint32_t> plan;
    plan.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(slot));
    const uint32_t offset = pair * SCORE_PLAN_LINE_WORDS;
    plan.SetValue(offset, begin.request);
    plan.SetValue(offset + 1U, begin.kMacro);
    plan.SetValue(offset + 2U, begin.qBand);
    plan.SetValue(offset + 3U, end.request);
    plan.SetValue(offset + 4U, end.kMacro);
    plan.SetValue(offset + 5U, end.qBand);
    WriteScoreCounter(plan, offset + 6U, candidate);
    WriteScoreCounter(plan, offset + 8U, direct);
    FlushScorePlanLine(plan, offset);
}

__aicore__ __attribute__((noinline)) static void SelectScorePlan(
    GM_ADDR slot, uint32_t pairs, ScoreWorkCosts costs)
{
    GlobalTensor<uint32_t> plan;
    plan.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(slot));
    uint64_t direct = 0U;
    uint64_t candidate = 0U;
    for (uint32_t pair = 0U; pair < pairs; ++pair) {
        const uint32_t offset = pair * SCORE_PLAN_LINE_WORDS;
        FlushScorePlanLine(plan, offset);
        const uint64_t pairCandidate = ReadScoreCounter(plan, offset + 6U);
        const uint64_t pairDirect = ReadScoreCounter(plan, offset + 8U);
        candidate = pairCandidate > candidate ? pairCandidate : candidate;
        direct = pairDirect > direct ? pairDirect : direct;
    }
    const uint64_t publication = 4U * pairs;
    WriteScoreCut(plan, pairs + 1U,
        {direct > candidate + publication + ScoreCutBoundaryCost(costs), 0U, 0U});
}

__aicore__ inline ScorePartition ReadScorePartition(GM_ADDR slot, uint32_t pairs, uint32_t pair)
{
    if (ReadScoreCut(slot, pairs + 1U).request == 0U) {
        return {};
    }
    GlobalTensor<uint32_t> plan;
    plan.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(slot));
    const uint32_t offset = pair * SCORE_PLAN_LINE_WORDS;
    FlushScorePlanLine(plan, offset);
    return {{plan.GetValue(offset), plan.GetValue(offset + 1U), plan.GetValue(offset + 2U)},
        {plan.GetValue(offset + 3U), plan.GetValue(offset + 4U), plan.GetValue(offset + 5U)}, true};
}

template <bool CAUSAL>
__aicore__ inline ScorePartition PrepareScorePartition(const PagedTilingData &tiling, GM_ADDR cu,
    GM_ADDR kv, GM_ADDR workspace, uint32_t qBand, uint32_t pair, LocalTensor<int32_t> metadata)
{
    const ScoreWorkCosts costs = GetScoreWorkCosts(tiling, qBand);
    if (tiling.numRequests == 1U) {
        return MakeSingleRequestPartition<CAUSAL>(tiling, cu, kv, qBand, pair, costs);
    }
    const uint32_t pairs = tiling.usedCubeCores;
    const uint32_t slots = IndexerRegionScore::ScorePlanWindowSlots(tiling.totalTokens, tiling.batchTokens, pairs);
    GM_ADDR summaries = workspace + ScoreCutSlotBytes(pairs) * slots;
    GM_ADDR descriptors = summaries + static_cast<uint64_t>(pairs) * IndexerRegionScore::SCORE_PLAN_LINE_BYTES;
    const uint32_t epoch = tiling.batchTokens == 0U ? 0U : tiling.batchBegin / tiling.batchTokens;
    const uint32_t slot = epoch % slots;
    if (epoch == 0U) {
        if ASCEND_IS_AIV {
            if (GetSubBlockIdx() == 0U) {
                PrepareScoreRequests<CAUSAL>(tiling, cu, kv, descriptors, summaries, qBand, pair, costs, metadata);
            }
        }
        RegionScoreSyncPolicy::GlobalBarrier();
    }
    if (slot == 0U) {
        uint32_t active = slots;
        if (tiling.batchTokens != 0U) {
            active = PagedMin(active, SparseIndexer::Common::CeilDiv(
                tiling.totalTokens - tiling.batchBegin, tiling.batchTokens));
        }
        if ASCEND_IS_AIV {
            if (GetSubBlockIdx() == 0U) {
                ScoreRequestTable table;
                table.Init(tiling, descriptors, summaries, metadata);
                for (uint32_t index = 0U; index < active; ++index) {
                    PagedTilingData planned = tiling;
                    if (tiling.batchTokens != 0U) {
                        planned.batchBegin += index * tiling.batchTokens;
                        planned.batchCount = PagedMin(tiling.batchTokens, tiling.totalTokens - planned.batchBegin);
                    }
                    WriteScorePairPlan<CAUSAL>(planned, cu, workspace + ScoreCutSlotBytes(pairs) * index,
                        qBand, costs, table, pair);
                }
            }
        }
        RegionScoreSyncPolicy::GlobalBarrier();
        if ASCEND_IS_AIV {
            if (GetSubBlockIdx() == 0U && pair < active) {
                SelectScorePlan(workspace + ScoreCutSlotBytes(pairs) * pair, pairs, costs);
            }
        }
        RegionScoreSyncPolicy::GlobalBarrier();
    }
    return ReadScorePartition(workspace + ScoreCutSlotBytes(pairs) * slot, pairs, pair);
}

} // namespace IndexerRegionScoreWeighted
#endif
