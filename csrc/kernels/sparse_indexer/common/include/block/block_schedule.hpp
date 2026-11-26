/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"

namespace SparseIndexer::Common::Block {

namespace detail {

template <class T, class U>
CATLASS_DEVICE constexpr uint32_t CeilDiv(const T& value, const U& divisor)
{
    return divisor == 0 ? 0 : static_cast<uint32_t>((value + divisor - 1) / divisor);
}

} // namespace detail

/**
 * Generic scheduler over a variant-owned schedule policy. The policy owns the
 * cursor state and the mapping to its own Coord type; the scheduler never
 * guesses a pipeline-specific work item.
 */
template <class SchedulePolicy_>
class BlockScheduler {
public:
    using SchedulePolicy = SchedulePolicy_;
    using State = typename SchedulePolicy::State;
    using Coord = typename SchedulePolicy::Coord;

    CATLASS_DEVICE
    explicit BlockScheduler(State const& state) : state_(state)
    {}

    template <class... Sources>
    CATLASS_DEVICE bool Next(Coord& coord, Sources&... sources)
    {
        return SchedulePolicy::Next(state_, coord, sources...);
    }

private:
    State state_{};
};

/**
 * Common request-source scheduler. The request ranges come from a
 * variant-specific source; the schedule policy owns Coord and may differ per
 * indexer variant.
 */
template <class RequestRangeSource_, class SchedulePolicy_>
class BlockSchedule {
public:
    using RequestRangeSource = RequestRangeSource_;
    using SchedulePolicy = SchedulePolicy_;
    using Coord = typename SchedulePolicy::Coord;

    struct Params {
        typename SchedulePolicy::Params policy{};
        RequestRangeSource source{};
    };

    CATLASS_DEVICE
    explicit BlockSchedule(Params const& params)
        : source_(params.source), policy_(params.policy)
    {
        totalMmadTasks_ = SchedulePolicy::CountMmad(source_, policy_);
        nextMmadTask_ = block_idx;
    }

    CATLASS_DEVICE
    bool Next(Coord& coord)
    {
        if (nextMmadTask_ >= totalMmadTasks_) {
            return false;
        }
        const uint64_t taskId = nextMmadTask_;
        nextMmadTask_ += block_num;
        return SchedulePolicy::FindMmad(source_, policy_, taskId, coord);
    }

    CATLASS_DEVICE
    RequestRangeSource const& source() const
    {
        return source_;
    }

    CATLASS_DEVICE
    typename SchedulePolicy::Params const& policy() const
    {
        return policy_;
    }

private:
    RequestRangeSource source_{};
    typename SchedulePolicy::Params policy_{};
    uint64_t totalMmadTasks_{0};
    uint64_t nextMmadTask_{0};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP
