#ifndef INDEXER_REGION_SCORE_BLOCK_REDUCE_HEAD_HPP
#define INDEXER_REGION_SCORE_BLOCK_REDUCE_HEAD_HPP

#include "stage_registry.hpp"

namespace IndexerRegionScorePaged {

struct RegionScoreBlockReduceHead {
    template <class Element>
    __aicore__ inline static void LoadWeightRows(
        LocalTensor<Element> destination, GlobalTensor<Element> source,
        uint32_t heads, uint32_t rows, uint64_t sourceRowStride)
    {
        // DataCopyPad rounds each destination row to one 32-byte boundary.
        const uint32_t rowBytes = heads * sizeof(Element);
        const uint32_t sourceGap = (sourceRowStride - heads) * sizeof(Element);
        const DataCopyExtParams params{static_cast<uint16_t>(rows), rowBytes, sourceGap, 0U, 0U};
        const DataCopyPadExtParams<Element> padding{false, 0U, 0U, Element{}};
        DataCopyPad(destination, source, params, padding);
    }

    __aicore__ inline static void LoadWeights(
        uint32_t dtype, uint32_t heads, uint32_t rows, uint64_t sourceRowStride,
        uint64_t sourceOffset, LocalTensor<float> fp32, LocalTensor<bfloat16_t> bf16,
        LocalTensor<half> fp16, GlobalTensor<float> fp32Gm,
        GlobalTensor<bfloat16_t> bf16Gm, GlobalTensor<half> fp16Gm)
    {
        if (dtype == 0U) {
            LoadWeightRows(fp32, fp32Gm[sourceOffset], heads, rows, sourceRowStride);
        } else if (dtype == 1U) {
            LoadWeightRows(bf16, bf16Gm[sourceOffset], heads, rows, sourceRowStride);
        } else {
            LoadWeightRows(fp16, fp16Gm[sourceOffset], heads, rows, sourceRowStride);
        }
    }

    __aicore__ inline static void ConvertWeights(
        uint32_t dtype, uint32_t heads, uint32_t rows, float scale,
        uint32_t fp32Stride, uint32_t nativeStride, LocalTensor<float> fp32,
        LocalTensor<bfloat16_t> bf16, LocalTensor<half> fp16)
    {
        for (uint32_t row = 0U; row < rows; ++row) {
            if (dtype == 1U) {
                Cast(fp32[row * fp32Stride], bf16[row * nativeStride],
                     RoundMode::CAST_NONE, heads);
            } else if (dtype == 2U) {
                Cast(fp32[row * fp32Stride], fp16[row * nativeStride],
                     RoundMode::CAST_NONE, heads);
            }
            if (scale != 1.0F) {
                Muls(fp32[row * fp32Stride], fp32[row * fp32Stride], scale, heads);
            }
        }
    }

    __aicore__ inline static bool CanUseOutput2D(uint64_t rowStride, uint32_t rowElements)
    {
        return rowStride >= rowElements &&
               (rowStride - rowElements) <=
                   static_cast<uint64_t>(UINT32_MAX) / sizeof(float);
    }

    __aicore__ inline static void StoreRows(const PagedTilingData &tiling, TEventID ready,
                                      GlobalTensor<float> &output, uint32_t group, uint32_t firstQ,
                                      uint32_t kStart, uint32_t kCount,
                                      uint32_t rowCount, uint32_t rowStride,
                                      LocalTensor<float> source)
    {
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        const uint64_t outputOffset =
            static_cast<uint64_t>(group) * tiling.outputStrideGroup +
            static_cast<uint64_t>(firstQ - tiling.batchBegin) * tiling.outputStrideToken + kStart;
        if (CanUseOutput2D(tiling.outputStrideToken, kCount)) {
            DataCopyExtParams params{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(kCount * sizeof(float)),
                static_cast<uint32_t>((rowStride - kCount) * sizeof(float) / 32U),
                static_cast<uint32_t>((tiling.outputStrideToken - kCount) * sizeof(float)), 0U};
            DataCopyPad(output[outputOffset], source, params);
            return;
        }
        for (uint32_t row = 0U; row < rowCount; ++row) {
            DataCopyExtParams params{
                1U, static_cast<uint32_t>(kCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(
                output[outputOffset + static_cast<uint64_t>(row) * tiling.outputStrideToken],
                source[row * rowStride], params);
        }
    }

};

} // namespace IndexerRegionScorePaged


namespace IndexerRegionScore::Block {
template <class KeyAdapter, bool QueryResident>
struct StageImplementation<Stage::ReduceHead, ComputeStagePolicy<KeyAdapter, QueryResident>> {
    using Type = IndexerRegionScorePaged::RegionScoreBlockReduceHead;
};
} // namespace IndexerRegionScore::Block

#endif
