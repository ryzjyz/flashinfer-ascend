#ifndef INDEXER_REGION_SCORE_BLOCK_MMAD_HPP
#define INDEXER_REGION_SCORE_BLOCK_MMAD_HPP

#include "stage_registry.hpp"

namespace IndexerRegionScorePaged {

struct RegionScoreBlockMmad {
    __aicore__ inline static void RunTileMmad(
        const PagedTilingData &tiling, LocalTensor<float> &cL0, LocalTensor<QkInputType> &qL0,
        LocalTensor<QkInputType> &kL0, TEventID fixToM, TEventID mToMte1, TEventID mToFix,
        uint32_t mAlign, uint32_t nAlign, uint32_t resultSlot)
    {
        WaitFlag<HardEvent::FIX_M>(fixToM);
        MmadParams mm;
        mm.m = mAlign;
        mm.n = nAlign;
        mm.k = QkInputDim(tiling);
        mm.cmatrixInitVal = true;
        mm.cmatrixSource = false;
        Mmad(cL0[resultSlot * PAGED_M_MAX * PAGED_NR],
             qL0[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_M_MAX * PAGED_D],
             kL0[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_NR * PAGED_D], mm);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(mToMte1);
        SetFlag<HardEvent::M_FIX>(mToFix);
        WaitFlag<HardEvent::M_FIX>(mToFix);
    }

    __aicore__ inline static void ConfigureCubeFixpipe(uint32_t heads,
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> &fp,
        uint32_t mAlign, uint32_t nAlign, uint32_t qCount,
        bool &singleAiv, uint32_t &firstRows, uint32_t &secondRows)
    {
        fp.nSize = nAlign;
        const uint32_t totalRows = mAlign;
        const uint32_t queryAlignment = heads % 16U == 0U ? 1U :
                                        heads % 8U == 0U ? 2U : 4U;
        const uint32_t alignedCapacity = ((PAGED_M_MAX / 2U) / heads) /
                                         queryAlignment * queryAlignment;
        singleAiv = totalRows <= PAGED_M_MAX / 2U || alignedCapacity == 0U;
        firstRows = singleAiv
                        ? PagedMin(PAGED_M_MAX / 2U, totalRows)
                        : PagedMin(qCount, alignedCapacity) * heads;
        secondRows = totalRows - firstRows;
        fp.mSize = firstRows;
        fp.srcStride = mAlign;
        fp.dstStride = PAGED_NR;
        fp.params.ndNum = 1U;
        fp.params.srcNdStride = 0U;
        fp.params.dstNdStride = 0U;
        fp.dualDstCtl = 0U;
        fp.reluEn = true;
        fp.subBlockId = 0U;
    }

    __aicore__ inline static void PublishCubeResult(
        uint32_t heads, LocalTensor<float> &cL0, LocalTensor<float> mmOut, TEventID fixToM,
        uint32_t mAlign, uint32_t nAlign, uint32_t qCount, uint32_t kCount,
        uint32_t resultSlot)
    {
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_FREE_EVENT + resultSlot);
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_C_FREE_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fp;
        bool singleAiv = false;
        uint32_t firstRows = 0U;
        uint32_t secondRows = 0U;
        ConfigureCubeFixpipe(heads, fp, mAlign, nAlign, qCount, singleAiv, firstRows, secondRows);
        Fixpipe<float, float, PAGED_FIX_TO_UB_ROW_MAJOR>(
            mmOut,
            cL0[resultSlot * PAGED_M_MAX * PAGED_NR], fp);
        if (!singleAiv) {
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_READY_EVENT + resultSlot);
        }
        if (secondRows != 0U) {
            fp.mSize = secondRows;
            fp.subBlockId = singleAiv ? 0U : 1U;
            Fixpipe<float, float, PAGED_FIX_TO_UB_ROW_MAJOR>(
                singleAiv
                    ? mmOut[firstRows * PAGED_NR]
                    : mmOut,
                cL0[resultSlot * PAGED_M_MAX * PAGED_NR + firstRows * 16U], fp);
        }
        if (singleAiv) {
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_READY_EVENT + resultSlot);
        }
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_C_READY_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);
        SetFlag<HardEvent::FIX_M>(fixToM);
    }
};

} // namespace IndexerRegionScorePaged


namespace IndexerRegionScore::Block {
template <class KeyAdapter, bool QueryResident>
struct StageImplementation<Stage::Mmad, ComputeStagePolicy<KeyAdapter, QueryResident>> {
    using Type = IndexerRegionScorePaged::RegionScoreBlockMmad;
};
} // namespace IndexerRegionScore::Block

#endif
