/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP

#include <cstdint>

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "contracts/common.hpp"

namespace SparseIndexer::Common::Block {

/**
 * Row-oriented TopK skeleton. The variant policy owns input preparation,
 * selection specialization, index finalization, and scratch layout; this
 * block owns AIV row partitioning and output-store lifecycle.
 */
template <class Policy_>
class BlockTopK {
public:
    using Policy = Policy_;
    using DtypeScore = typename Policy::DtypeScore;
    using DtypeOut = typename Policy::DtypeOut;
    using ArchTag = Catlass::Arch::Ascend950;
    using Params = typename Policy::Params;
    using Scratch = typename Policy::Scratch;
    using VectorBlock = BlockTopK;

    static constexpr uint32_t kSplitS1 = Policy::kSplitS1;

    CATLASS_DEVICE BlockTopK(Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<DtypeScore> const& workspace, AscendC::GlobalTensor<DtypeOut> const& outIndices,
        Params const& params, AscendC::GlobalTensor<int32_t> const& outputIdxOffset)
        : gmScores_(workspace),
          gmOut_(outIndices),
          gmOutputIdxOffset_(outputIdxOffset),
          params_(params)
    {
        Policy::InitScratch(scratch_, resource, params);
    }

    CATLASS_DEVICE __attribute__((always_inline)) void operator()(uint32_t s1Start, uint32_t actualS1,
        uint32_t kStart, uint32_t kLen,
        uint32_t localQStart, uint32_t qLen, uint32_t requestKLen, int32_t residual)
    {
        if (actualS1 == 0) {
            return;
        }
        uint32_t rowBegin = 0;
        uint32_t rowCount = actualS1;
        if constexpr (!Policy::kLocalRows) {
            const uint32_t rowsPerSub = SparseIndexer::Common::CeilDiv(actualS1, kSplitS1);
            const uint32_t sub = AscendC::GetSubBlockIdx();
            rowBegin = sub * rowsPerSub;
            if (rowBegin >= actualS1) {
                return;
            }
            rowCount = AscendC::Std::min(rowsPerSub, actualS1 - rowBegin);
        }

        AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(params_.topK * sizeof(DtypeOut)), 0, 0, 0};
        AscendC::LocalTensor<DtypeOut> outIdx = scratch_.outIdxUb.template ReinterpretCast<DtypeOut>();

        for (uint32_t ri = 0; ri < rowCount; ++ri) {
            const uint32_t s1Row = s1Start + rowBegin + ri;
            const uint32_t visibleLen = Policy::VisibleLen(
                localQStart + rowBegin + ri, qLen, requestKLen, residual);
            if (visibleLen < params_.topK) {
                Policy::BuildPrefix(scratch_, visibleLen, params_.topK);
            } else {
                Policy::Select(scratch_, gmScores_, s1Row, kStart, kLen, visibleLen, params_);
            }

            int32_t rowOff = Policy::OutputRowOffset(gmOutputIdxOffset_, s1Row, params_);
            AscendC::PipeBarrier<PIPE_V>();
            Policy::Finalize(scratch_, outIdx, visibleLen, params_.topK,
                static_cast<int32_t>(kStart) + rowOff);
            if constexpr (Policy::kLocalRows) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            storeRow(outIdx, s1Row, copyOut);
        }
    }

private:
    CATLASS_DEVICE void storeRow(AscendC::LocalTensor<DtypeOut> const& outIdx,
        uint32_t s1Row, AscendC::DataCopyExtParams const& copyOut)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmOut_[s1Row * params_.outputRowStride], outIdx, copyOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    AscendC::GlobalTensor<DtypeScore> const& gmScores_;
    AscendC::GlobalTensor<DtypeOut> const& gmOut_;
    AscendC::GlobalTensor<int32_t> const& gmOutputIdxOffset_;
    Scratch scratch_{};
    Params params_{};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP
