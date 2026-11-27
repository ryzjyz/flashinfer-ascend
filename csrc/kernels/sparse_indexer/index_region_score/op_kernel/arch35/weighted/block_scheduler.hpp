#ifndef IRS_WEIGHTED_BLOCK_SCHEDULER_HPP
#define IRS_WEIGHTED_BLOCK_SCHEDULER_HPP

#include "block/block_schedule.hpp"
#include "device_score_plan.hpp"

namespace IndexerRegionScoreWeighted {

struct RegionScoreBatchPolicy {
    struct State {
        uint32_t totalTokens;
        uint32_t batchTokens;
        uint32_t nextToken = 0U;
    };
    struct Coord {
        uint32_t begin;
        uint32_t count;
    };

    __aicore__ static inline bool Next(State &state, Coord &coord)
    {
        if (state.batchTokens == 0U || state.nextToken >= state.totalTokens) {
            return false;
        }
        coord.begin = state.nextToken;
        coord.count = PagedMin(state.batchTokens, state.totalTokens - state.nextToken);
        state.nextToken += coord.count;
        return true;
    }
};

struct RegionScoreKeyMacroPolicy {
    struct State {
        uint32_t nextKey;
        uint32_t keyEnd;
        uint32_t nextIndex;
    };
    struct Coord {
        uint32_t start;
        uint32_t count;
        uint32_t index;
    };

    __aicore__ static inline bool Next(State &state, Coord &coord)
    {
        if (state.nextKey >= state.keyEnd) {
            return false;
        }
        coord = {state.nextKey, PagedMin(PAGED_K_MACRO, state.keyEnd - state.nextKey), state.nextIndex};
        state.nextKey += PAGED_K_MACRO;
        ++state.nextIndex;
        return true;
    }
};

template <bool CAUSAL, bool QUERY_RESIDENT>
struct RegionScoreRequestPolicy {
    struct WorkRange {
        uint32_t qBegin;
        uint32_t qEnd;
        uint32_t kBegin;
        uint32_t kEnd;
    };
    struct State {
        const PagedTilingData &tiling;
        uint32_t core;
        WorkRange range;
        uint32_t nextRequest = 0U;
        uint32_t requestKCarry = 0U;
        uint32_t requestLimit = 0U;
        bool weighted = false;
        DeviceScoreCut cursor{};
        DeviceScoreCut finish{};
        DeviceWorkShape shape{};
        uint32_t requestBegin = 0U;
        bool prepared = false;
    };
    struct Coord {
        uint32_t request;
        uint32_t requestQBegin;
        uint32_t qBegin;
        uint32_t qEnd;
        uint32_t kBegin;
        uint32_t kEnd;
    };

    __aicore__ static inline State Init(const PagedTilingData &tiling, uint32_t core)
    {
        State state{tiling, core, GetCoreWorkRange(tiling, core)};
        state.requestLimit = tiling.numRequests;
        return state;
    }

    __aicore__ static inline void RestrictRequests(State &state, GM_ADDR cu)
    {
        if (state.range.qBegin >= state.range.qEnd) {
            state.requestLimit = 0U;
            return;
        }
        if (state.tiling.numRequests > 1U) {
            state.nextRequest = LocateScoreRequest(cu, state.tiling.numRequests, state.range.qBegin);
            state.requestLimit = PagedMin(
                LocateScoreRequest(cu, state.tiling.numRequests, state.range.qEnd - 1U) + 1U,
                state.tiling.numRequests);
        }
    }

    __aicore__ static inline void UseWeightedRange(State &state, const ScorePartition &partition)
    {
        state.weighted = true;
        state.cursor = partition.begin;
        state.finish = partition.end;
    }

    template <class RequestSource>
    __aicore__ static inline bool Next(State &state, Coord &coord, RequestSource &source)
    {
        if (state.weighted) {
            return NextWeighted(state, coord, source);
        }
        while (state.nextRequest < state.requestLimit) {
            const uint32_t request = state.nextRequest++;
            uint32_t requestQBegin = 0U;
            uint32_t requestQEnd = 0U;
            if (!source.template LoadRequestSpan<CAUSAL>(request, requestQBegin, requestQEnd)) {
                continue;
            }
            const uint32_t qBegin = PagedMax(requestQBegin, state.range.qBegin);
            const uint32_t qEnd = PagedMin(requestQEnd, state.range.qEnd);
            if (qBegin >= qEnd) {
                continue;
            }
            uint32_t kBegin = state.range.kBegin;
            uint32_t kEnd = state.range.kEnd;
            if constexpr (CAUSAL) {
                // Every participant in this CV pair uses the same last Q token.
                const auto visible = static_cast<uint32_t>(source.VisibleRegions(request, requestQBegin, qEnd - 1U));
                if (!AssignCausalKeyRange(state, visible, kBegin, kEnd)) {
                    continue;
                }
            }
            coord = {request, requestQBegin, qBegin, qEnd, kBegin, kEnd};
            return true;
        }
        return false;
    }

private:
    __aicore__ static inline bool AtFinish(const State &state)
    {
        const auto &cur = state.cursor;
        const auto &end = state.finish;
        return cur.request > end.request ||
            (cur.request == end.request && (cur.kMacro > end.kMacro ||
                (cur.kMacro == end.kMacro && cur.qBand >= end.qBand)));
    }

    __aicore__ static inline void AdvanceWeightedRequest(State &state)
    {
        ++state.cursor.request;
        state.cursor.kMacro = 0U;
        state.cursor.qBand = 0U;
        state.prepared = false;
    }

    template <class RequestSource>
    __aicore__ __attribute__((always_inline)) static inline bool NextWeighted(
        State &state, Coord &coord, RequestSource &source)
    {
        const auto &tiling = state.tiling;
        const uint32_t epochBegin = tiling.batchTokens == 0U ? 0U : tiling.batchBegin;
        const uint32_t epochEnd = tiling.batchTokens == 0U
            ? tiling.totalTokens : tiling.batchBegin + tiling.batchCount;
        while (state.cursor.request < tiling.numRequests && !AtFinish(state)) {
            const uint32_t rid = state.cursor.request;
            if (!state.prepared) {
                uint32_t requestEnd = 0U;
                if (!source.template LoadRequestSpan<CAUSAL>(rid, state.requestBegin, requestEnd)) {
                    AdvanceWeightedRequest(state);
                    continue;
                }
                const uint32_t qBegin = PagedMax(state.requestBegin, epochBegin);
                const uint32_t qEnd = PagedMin(requestEnd, epochEnd);
                if (qBegin >= qEnd) {
                    AdvanceWeightedRequest(state);
                    continue;
                }
                int64_t position = 0;
                if constexpr (CAUSAL) {
                    position = source.requestFirstPosition_ + qBegin - state.requestBegin;
                }
                state.shape = {qBegin, qEnd, source.balanceQBand_, PAGED_K_MACRO, tiling.seqK,
                               tiling.regionSize, position, CAUSAL};
                state.prepared = true;
            }
            const auto &shape = state.shape;
            const uint32_t k = state.cursor.kMacro;
            const uint32_t kTotal = KMacroCount(shape);
            if (k >= kTotal) {
                AdvanceWeightedRequest(state);
                continue;
            }

            const uint32_t bands = BandCount(shape);
            const uint32_t firstActive = FirstActiveBand(shape, k);
            const uint32_t qFirst = PagedMax(firstActive, state.cursor.qBand);
            const bool endsAtThisK = state.finish.request == rid && state.finish.kMacro == k;
            const uint32_t qStop = endsAtThisK ? PagedMin(state.finish.qBand, bands) : bands;
            if (qFirst >= qStop) {
                ++state.cursor.kMacro;
                state.cursor.qBand = 0U;
                continue;
            }

            uint32_t runEnd = k + 1U;
            if (qFirst == firstActive && qStop == bands && !endsAtThisK) {
                // A plateau of active Q bands allows the existing K prefetch and Q reuse.
                const uint32_t limit = state.finish.request == rid
                    ? PagedMin(kTotal, state.finish.kMacro) : kTotal;
                runEnd = PagedMin(limit, BandKEnd(shape, firstActive));
            }
            coord = {rid, state.requestBegin,
                shape.qBegin + qFirst * shape.qBand,
                PagedMin(shape.qEnd, shape.qBegin + qStop * shape.qBand),
                k * PAGED_K_MACRO,
                PagedMin(tiling.seqK, runEnd * PAGED_K_MACRO)};
            state.cursor.kMacro = runEnd;
            state.cursor.qBand = 0U;
            return true;
        }
        return false;
    }

    __aicore__ static inline WorkRange GetCoreWorkRange(const PagedTilingData &tiling, uint32_t core)
    {
        const uint32_t rangeUnit = tiling.qTile;
        const uint32_t totalQRanges =
            ((tiling.batchTokens != 0U ? tiling.batchCount : tiling.totalTokens) + rangeUnit - 1U) / rangeUnit;
        const uint32_t totalKRanges =
            (tiling.seqK + tiling.kTile - 1U) / tiling.kTile;
        const uint32_t qShard = core / tiling.kShardCount;
        const uint32_t kShard = core % tiling.kShardCount;

        const uint32_t qRangesPerShard = totalQRanges / tiling.qShardCount;
        const uint32_t qExtra = totalQRanges % tiling.qShardCount;
        const uint32_t firstQRange =
            qShard * qRangesPerShard + PagedMin(qShard, qExtra);
        const uint32_t localQRanges =
            qRangesPerShard + (qShard < qExtra ? 1U : 0U);
        const uint32_t qBegin = tiling.batchBegin + firstQRange * rangeUnit;
        const uint32_t qEnd = tiling.batchBegin + PagedMin(
            tiling.batchTokens != 0U ? tiling.batchCount : tiling.totalTokens,
            (firstQRange + localQRanges) * rangeUnit);

        const uint32_t kRangesPerShard = totalKRanges / tiling.kShardCount;
        const uint32_t kExtra = totalKRanges % tiling.kShardCount;
        const uint32_t firstKRange =
            kShard * kRangesPerShard + PagedMin(kShard, kExtra);
        const uint32_t localKRanges =
            kRangesPerShard + (kShard < kExtra ? 1U : 0U);
        return {qBegin, qEnd, firstKRange * tiling.kTile,
                PagedMin(tiling.seqK, (firstKRange + localKRanges) * tiling.kTile)};
    }

    __aicore__ static inline bool AssignCausalKeyRange(
        State &state, uint32_t visible, uint32_t &kBegin, uint32_t &kEnd)
    {
        const auto &tiling = state.tiling;
        const uint32_t macroCount = (visible + PAGED_K_MACRO - 1U) / PAGED_K_MACRO;
        if constexpr (!QUERY_RESIDENT) {
            if (macroCount == (tiling.seqK + PAGED_K_MACRO - 1U) / PAGED_K_MACRO) {
                return kBegin < kEnd;
            }
        }
        const uint32_t physicalKShard = state.core % tiling.kShardCount;
        const uint32_t kShard = (physicalKShard + tiling.kShardCount - state.requestKCarry) % tiling.kShardCount;
        const uint32_t macrosPerShard = macroCount / tiling.kShardCount;
        const uint32_t extra = macroCount % tiling.kShardCount;
        // Advance the rotation even when this CV pair receives no macros.
        state.requestKCarry = (state.requestKCarry + extra) % tiling.kShardCount;
        const uint32_t firstMacro = kShard * macrosPerShard + PagedMin(kShard, extra);
        const uint32_t localMacros = macrosPerShard + (kShard < extra ? 1U : 0U);
        kBegin = firstMacro * PAGED_K_MACRO;
        kEnd = PagedMin(tiling.seqK, (firstMacro + localMacros) * PAGED_K_MACRO);
        return kBegin < kEnd;
    }
};

} // namespace IndexerRegionScoreWeighted

#endif
