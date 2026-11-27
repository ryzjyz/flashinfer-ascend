#ifndef INDEXER_REGION_SCORE_DETAIL_HEAD_REDUCE_VF_HPP
#define INDEXER_REGION_SCORE_DETAIL_HEAD_REDUCE_VF_HPP

#include "kernel_operator.h"

namespace IndexerRegionScoreVector {

// Four-head chunks preserve the FP32 rounding order of the original reduction.
template <uint32_t SEGMENTS>
__simd_callee__ inline void ReduceFourHeadVectors(
    __ubuf__ float *source, __ubuf__ float *weights,
    AscendC::MicroAPI::MaskReg &firstMask,
    AscendC::MicroAPI::MaskReg &secondMask,
    AscendC::MicroAPI::RegTensor<float> &first,
    AscendC::MicroAPI::RegTensor<float> &second)
{
    using namespace AscendC;
    MicroAPI::RegTensor<float> inputFirst, inputSecond, weight;
    MicroAPI::LoadAlign(inputFirst, source);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(weight, weights);
    MicroAPI::Mul(first, inputFirst, weight, firstMask);
    if constexpr (SEGMENTS == 2U) {
        MicroAPI::LoadAlign(inputSecond, source + 64U);
        MicroAPI::Mul(second, inputSecond, weight, secondMask);
    }

    for (uint16_t head = 1U; head < 4U; ++head) {
        MicroAPI::LoadAlign(inputFirst, source + head * 128U);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(weight, weights + head);
        MicroAPI::MulAddDst(first, inputFirst, weight, firstMask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(inputSecond, source + head * 128U + 64U);
            MicroAPI::MulAddDst(second, inputSecond, weight, secondMask);
        }
    }
}

template <uint32_t SEGMENTS, bool CAUSAL>
__simd_vf__ inline void WeightedReduceTileVFImpl(
    __ubuf__ float *destination, __ubuf__ float *source,
    __ubuf__ float *weights, __ubuf__ int32_t *visible,
    uint32_t rows, uint32_t heads, uint32_t weightStride,
    uint32_t columns, int32_t tileBegin)
{
    using namespace AscendC;
    static_assert(SEGMENTS == 1U || SEGMENTS == 2U,
                  "head reduction accepts one or two 64-column segments");
    uint32_t firstColumns = columns < 64U ? columns : 64U;
    uint32_t secondColumns = SEGMENTS == 2U ? columns - 64U : 0U;
    MicroAPI::MaskReg firstMask = MicroAPI::UpdateMask<float>(firstColumns);
    MicroAPI::MaskReg secondMask = MicroAPI::UpdateMask<float>(secondColumns);
    MicroAPI::RegTensor<int32_t> firstColumnIds, secondColumnIds, visibleLength;
    MicroAPI::RegTensor<uint32_t> negativeInfinityBits;
    if constexpr (CAUSAL) {
        MicroAPI::Arange(firstColumnIds, tileBegin);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::Arange(secondColumnIds, tileBegin + 64);
        }
        MicroAPI::MaskReg allLanes = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(negativeInfinityBits, 0xFF800000U, allLanes);
    }

    for (uint16_t row = 0U; row < rows; ++row) {
        __ubuf__ float *rowSource = source + row * heads * 128U;
        __ubuf__ float *rowWeights = weights + row * weightStride;
        MicroAPI::RegTensor<float> firstTotal, secondTotal, firstPartial, secondPartial;
        ReduceFourHeadVectors<SEGMENTS>(
            rowSource, rowWeights, firstMask, secondMask, firstTotal, secondTotal);
        for (uint16_t head = 4U; head < heads; head += 4U) {
            ReduceFourHeadVectors<SEGMENTS>(
                rowSource + head * 128U, rowWeights + head,
                firstMask, secondMask, firstPartial, secondPartial);
            MicroAPI::Add(firstTotal, firstTotal, firstPartial, firstMask);
            if constexpr (SEGMENTS == 2U) {
                MicroAPI::Add(secondTotal, secondTotal, secondPartial, secondMask);
            }
        }

        if constexpr (CAUSAL) {
            MicroAPI::MaskReg firstVisible, secondVisible;
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(
                visibleLength, visible + row);
            MicroAPI::Compare<int32_t, CMPMODE::LT>(
                firstVisible, firstColumnIds, visibleLength, firstMask);
            MicroAPI::Select(firstTotal, firstTotal,
                reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), firstVisible);
            if constexpr (SEGMENTS == 2U) {
                MicroAPI::Compare<int32_t, CMPMODE::LT>(
                    secondVisible, secondColumnIds, visibleLength, secondMask);
                MicroAPI::Select(secondTotal, secondTotal,
                    reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), secondVisible);
            }
        }
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            destination + row * 128U, firstTotal, firstMask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                destination + row * 128U + 64U, secondTotal, secondMask);
        }
    }
}

// The caller owns events and GM stores. MMAD and Score UB use 128-column rows.
template <bool CAUSAL>
__aicore__ inline void WeightedReduceTileVF(
    const AscendC::LocalTensor<float> &destination,
    const AscendC::LocalTensor<float> &source,
    const AscendC::LocalTensor<float> &weights,
    const AscendC::LocalTensor<int32_t> &visible,
    uint32_t rows, uint32_t heads, uint32_t weightStride,
    uint32_t columns, int32_t tileBegin)
{
    auto destinationPointer = reinterpret_cast<__ubuf__ float *>(destination.GetPhyAddr());
    auto sourcePointer = reinterpret_cast<__ubuf__ float *>(source.GetPhyAddr());
    auto weightPointer = reinterpret_cast<__ubuf__ float *>(weights.GetPhyAddr());
    auto visiblePointer = reinterpret_cast<__ubuf__ int32_t *>(visible.GetPhyAddr());
    if (columns <= 64U) {
        asc_vf_call<WeightedReduceTileVFImpl<1U, CAUSAL>>(
            destinationPointer, sourcePointer, weightPointer, visiblePointer,
            rows, heads, weightStride, columns, tileBegin);
    } else {
        asc_vf_call<WeightedReduceTileVFImpl<2U, CAUSAL>>(
            destinationPointer, sourcePointer, weightPointer, visiblePointer,
            rows, heads, weightStride, columns, tileBegin);
    }
}

} // namespace IndexerRegionScoreVector

#endif
