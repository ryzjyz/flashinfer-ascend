#ifndef INDEXER_REGION_SCORE_BLOCK_SCHEDULER_HPP
#define INDEXER_REGION_SCORE_BLOCK_SCHEDULER_HPP

#include "block/block_schedule.hpp"

namespace IndexerRegionScorePaged {

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
        return {tiling, core, GetCoreWorkRange(tiling, core)};
    }

    template <class RequestSource>
    __aicore__ static inline bool Next(State &state, Coord &coord, RequestSource &source)
    {
        while (state.nextRequest < state.tiling.numRequests) {
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

} // namespace IndexerRegionScorePaged

#endif
