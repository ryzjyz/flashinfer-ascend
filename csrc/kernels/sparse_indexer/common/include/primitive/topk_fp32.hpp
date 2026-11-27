/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_PRIMITIVE_TOPK_FP32_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_PRIMITIVE_TOPK_FP32_HPP

#include <climits>
#include <cstdint>
#include "kernel_operator.h"

namespace SparseIndexer::Common::Primitive::TopK {
using namespace AscendC;

// Hardware units and FP32 radix geometry are independent of caller problem sizes.
inline constexpr uint32_t kKeyBytes = sizeof(uint32_t);
inline constexpr uint32_t kKeyByteCount = sizeof(uint32_t) / sizeof(uint8_t);
inline constexpr uint32_t kByteBits = CHAR_BIT;
inline constexpr uint32_t kHistogramBins = 1U << kByteBits;
inline constexpr uint32_t kVectorBytes = AscendC::VECTOR_REG_WIDTH;
inline constexpr uint32_t kKeyLanes = kVectorBytes / kKeyBytes;
inline constexpr uint32_t kHalfwordLanes = kVectorBytes / sizeof(uint16_t);
inline constexpr uint32_t kHistogramInputElements = kVectorBytes / sizeof(uint8_t);
inline constexpr uint32_t kHistogramVectorCount = kHistogramBins / kKeyLanes;
inline constexpr uint32_t kStoreAlignmentBytes = AscendC::ONE_BLK_SIZE;
inline constexpr uint32_t kHighByteShift = (kKeyByteCount - 1U) * kByteBits;
inline constexpr uint32_t kSecondByteShift = (kKeyByteCount - 2U) * kByteBits;
inline constexpr uint32_t kKeySignBit = 1U << (sizeof(uint32_t) * CHAR_BIT - 1U);
inline constexpr uint32_t kAllKeyBits = UINT32_MAX;

template <uint32_t MaxElements = 16U * 1024U, uint32_t MaxKeepCount = 4U * 1024U>
struct HighPrefixWorkspace {
    static_assert(MaxElements > 0U && MaxElements % kHistogramInputElements == 0U);
    static_assert(MaxElements <= UINT16_MAX, "Radix histogram counters must not overflow");
    static_assert(MaxKeepCount > 0U && MaxKeepCount <= MaxElements);
    static constexpr uint32_t kMaxElements = MaxElements;
    static constexpr uint32_t kMaxKeepCount = MaxKeepCount;
    static constexpr uint32_t kPositionsOffset = kMaxElements * kKeyBytes;
    static constexpr uint32_t kHistogramOffset = kPositionsOffset + kMaxElements * kKeyBytes;
    static constexpr uint32_t kBucket0Offset = kHistogramBins;
    static constexpr uint32_t kBucket1Offset = kBucket0Offset + kHistogramBins;
    static constexpr uint32_t kBucket2Offset = kBucket1Offset + kHistogramBins;
    static constexpr uint32_t kBucket3Offset = kBucket2Offset + kHistogramBins;
    static constexpr uint32_t kRankOffset = kBucket3Offset + kHistogramBins;
    static constexpr uint32_t kCountOffset = kRankOffset + kKeyLanes;
    static constexpr uint32_t kWorkspaceBytes =
        kHistogramOffset + kCountOffset * kKeyBytes + kStoreAlignmentBytes;
};

using DefaultHighPrefixWorkspace = HighPrefixWorkspace<>;
inline constexpr uint32_t kMaxElements = DefaultHighPrefixWorkspace::kMaxElements;
inline constexpr uint32_t kMaxKeepCount = DefaultHighPrefixWorkspace::kMaxKeepCount;
inline constexpr uint32_t kWorkspaceBytes = DefaultHighPrefixWorkspace::kWorkspaceBytes;

// Convert caller-provided FP32 scores in place, excluding invalid values and inactive lanes.
__simd_vf__ void PrepareFp32Keys(__ubuf__ uint32_t *buffer, uint32_t validCount,
    uint32_t alignedCount, float invalidScore)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zeroBits;
    Reg::RegTensor<uint32_t> signBit;
    Reg::RegTensor<uint32_t> allBits;
    Reg::RegTensor<int32_t> validLimit;
    Reg::RegTensor<float> zeroFloat;
    Reg::RegTensor<float> validThreshold;
    Reg::Duplicate(zeroBits, 0U);
    Reg::Duplicate(signBit, kKeySignBit);
    Reg::Duplicate(allBits, kAllKeyBits);
    Reg::Duplicate(validLimit, static_cast<int32_t>(validCount));
    Reg::Duplicate(zeroFloat, 0.0f);
    Reg::Duplicate(validThreshold, invalidScore);
    for (uint32_t offset = 0U; offset < alignedCount; offset += kKeyLanes) {
        Reg::RegTensor<uint32_t> bits;
        Reg::RegTensor<int32_t> positions;
        Reg::RegTensor<uint32_t> sign;
        Reg::RegTensor<uint32_t> xorMask;
        Reg::RegTensor<uint32_t> key;
        Reg::MaskReg isZero;
        Reg::MaskReg isNegative;
        Reg::MaskReg inRange;
        Reg::MaskReg scoreValid;
        Reg::MaskReg keep;
        Reg::LoadAlign<uint32_t>(bits, buffer + offset);
        Reg::Arange(positions, static_cast<int32_t>(offset));
        Reg::Compare<float, CMPMODE::EQ>(isZero,
            reinterpret_cast<Reg::RegTensor<float> &>(bits), zeroFloat, all);
        Reg::Select(bits, zeroBits, bits, isZero);
        Reg::And(sign, bits, signBit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(isNegative, sign, zeroBits, all);
        Reg::Select(xorMask, allBits, signBit, isNegative);
        Reg::Xor(key, bits, xorMask, all);
        Reg::Compare<int32_t, CMPMODE::LT>(inRange, positions, validLimit, all);
        Reg::Compare<float, CMPMODE::GT>(scoreValid,
            reinterpret_cast<Reg::RegTensor<float> &>(bits), validThreshold, all);
        Reg::And(keep, inRange, scoreValid, all);
        Reg::Select(key, key, zeroBits, keep);
        Reg::StoreAlign<uint32_t>(buffer + offset, key, all);
    }
}

namespace RadixDetail {
using namespace AscendC;
// 保持直方图交错布局，转换计数后写出。
__simd_callee__ inline void StoreHistogram(__ubuf__ uint32_t *histogramsBuf,
    Reg::RegTensor<uint16_t> &cout0, Reg::RegTensor<uint16_t> &cout1,
    Reg::RegTensor<uint32_t> &cout0U32Even, Reg::RegTensor<uint32_t> &cout0U32Odd,
    Reg::RegTensor<uint32_t> &cout1U32Even, Reg::RegTensor<uint32_t> &cout1U32Odd,
    Reg::MaskReg &pregB16Reg, Reg::MaskReg &pregB32Reg)
{
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + kHalfwordLanes, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg);
}

template <bool LOAD_K, bool LAST = false>
__simd_callee__ inline void FindTargetBinInline(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *kValue, uint32_t bottomK,
    __ubuf__ uint32_t *idx0Buf = nullptr, __ubuf__ uint32_t *idx1Buf = nullptr, __ubuf__ uint32_t *idx2Buf = nullptr)
{
    Reg::MaskReg pregB32Reg = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> btmK;
    if constexpr (LOAD_K) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK, kValue);
    } else {
        Reg::Duplicate(btmK, bottomK);
    }
    for (uint16_t i = 0; i < kHistogramVectorCount; ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx;
        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Arange(idxC, i * kKeyLanes);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * kKeyLanes);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32Reg);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxBuf, sqzIdx, alignIdx);
    }
    Reg::StoreUnAlignPost(idxBuf, alignIdx);
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    if constexpr (LAST) {
        Reg::RegTensor<uint32_t> idx0, idx1, idx2, idx3;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx3, idxBuf);
        Reg::ShiftLefts(idx0, idx0, static_cast<int16_t>(kHighByteShift), pregB32Reg);
        Reg::ShiftLefts(idx1, idx1, static_cast<int16_t>(kSecondByteShift), pregB32Reg);
        Reg::ShiftLefts(idx2, idx2, static_cast<int16_t>(kByteBits), pregB32Reg);
        Reg::Add(idx0, idx0, idx1, pregB32Reg);
        Reg::Add(idx0, idx0, idx2, pregB32Reg);
        Reg::Add(idx0, idx0, idx3, pregB32Reg);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32Reg);
    } else {
        Reg::RegTensor<uint32_t> idx;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx, idxBuf);
        Reg::RegTensor<uint8_t> idxAll1;
        Reg::RegTensor<uint32_t> idxPrev;
        Reg::RegTensor<uint32_t> prevBinValue;
        Reg::Duplicate(idxAll1, 1);
        Reg::RegTensor<uint32_t> zeroAll;
        Reg::Duplicate(zeroAll, 0);
        Reg::MaskReg pregZero = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32Reg);
        Reg::Sub(idxPrev, idx, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32Reg);
        Reg::ShiftRights(idxPrev, idxPrev, static_cast<int16_t>(kHighByteShift), pregB32Reg);
        Reg::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32Reg);
        Reg::Select(prevBinValue, zeroAll, prevBinValue, pregZero);
        Reg::RegTensor<uint32_t> nextK;
        Reg::Sub(nextK, btmK, prevBinValue, pregB32Reg);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg);
    }
}

template <bool LOAD_K, bool LAST = false>
__simd_vf__ void FindTargetBinVFImpl(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
    __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *kValue, uint32_t bottomK,
    __ubuf__ uint32_t *idx0Buf = nullptr, __ubuf__ uint32_t *idx1Buf = nullptr,
    __ubuf__ uint32_t *idx2Buf = nullptr)
{
    FindTargetBinInline<LOAD_K, LAST>(idxBuf, nkValueBuf, histogramsBuf, kValue,
        bottomK, idx0Buf, idx1Buf, idx2Buf);
}

// Compile-time byte stages share the load, prefix mask and histogram layout.
template <uint32_t BYTE>
__simd_callee__ inline void HistogramsInline(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
    __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop)
{
    auto all32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    auto all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    auto all8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0Even, cout0Odd, cout1Even, cout1Odd;
    Reg::RegTensor<uint32_t> idx0, idx1, idx2;
    if constexpr (BYTE >= 1U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    if constexpr (BYTE >= 2U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    if constexpr (BYTE >= 3U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);
    Reg::RegTensor<uint32_t> lo0, hi0, lo1, hi1;
    Reg::RegTensor<uint8_t> byte0, byte1, byte2, byte3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(lo0, hi0, inputBuf + i * kHistogramInputElements);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(lo1, hi1, inputBuf + i * kHistogramInputElements + kHalfwordLanes);
        Reg::DeInterleave(byte1, byte0, (Reg::RegTensor<uint8_t> &)hi0, (Reg::RegTensor<uint8_t> &)hi1);
        if constexpr (BYTE >= 2U) {
            Reg::DeInterleave(byte3, byte2, (Reg::RegTensor<uint8_t> &)lo0, (Reg::RegTensor<uint8_t> &)lo1);
        }
        auto keep = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        if constexpr (BYTE >= 1U) {
            Reg::Compare<uint8_t, CMPMODE::EQ>(keep, byte0, (Reg::RegTensor<uint8_t> &)idx0, all8);
        }
        if constexpr (BYTE >= 2U) {
            Reg::MaskReg match;
            Reg::Compare<uint8_t, CMPMODE::EQ>(match, byte1, (Reg::RegTensor<uint8_t> &)idx1, all8);
            Reg::And(keep, keep, match, all8);
        }
        if constexpr (BYTE >= 3U) {
            Reg::MaskReg match;
            Reg::Compare<uint8_t, CMPMODE::EQ>(match, byte2, (Reg::RegTensor<uint8_t> &)idx2, all8);
            Reg::And(keep, keep, match, all8);
        }
        if constexpr (BYTE == 0U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte0, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte0, keep);
        }
        if constexpr (BYTE == 1U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte1, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte1, keep);
        }
        if constexpr (BYTE == 2U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte2, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte2, keep);
        }
        if constexpr (BYTE == 3U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte3, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte3, keep);
        }
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0Even, cout0Odd, cout1Even, cout1Odd, all16, all32);
}

template <uint32_t BYTE>
__simd_vf__ void HistogramsVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
    __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf,
    uint16_t vfLoop)
{
    HistogramsInline<BYTE>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop);
}
template <bool EQUAL>
__simd_vf__ void FindIdxOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf,
    __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    // The equality pass appends to the greater-than pass's squeeze count.
    if constexpr (!EQUAL) Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> kthValue, input, packed;
    Reg::RegTensor<int32_t> indices;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::MaskReg selected;
        Reg::Arange(indices, i * kKeyLanes);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(input, inputBuf + i * kKeyLanes);
        Reg::Compare<uint32_t, EQUAL ? CMPMODE::EQ : CMPMODE::GT>(selected, input, kthValue, all);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packed, (Reg::RegTensor<uint32_t> &)indices, selected);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, packed, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputValueBuf,
                                       __ubuf__ uint32_t *tmpIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg10 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputValue;
    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * kKeyLanes);
        Reg::Gather(outputValue, inputValueBuf, tmpIdx, pregB32Reg10);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputValueBuf + i * kKeyLanes, outputValue, pregB32Reg10);
    }
}

// Select indices from padded keys and optionally gather the selected values.
template <bool ISOUTVALUE>
__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &tmpIdxLocal,
                                const LocalTensor<uint32_t> &outputValueLocal,
                                const LocalTensor<uint32_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t validLen)
{
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();
    uint32_t bottomK = validLen - topK + 1;
    uint16_t histogramsLoopNum = (validLen + kHistogramInputElements - 1U) / kHistogramInputElements;
    uint16_t inputLoopNum = (validLen + kKeyLanes - 1U) / kKeyLanes;
    HistogramsVFImpl<0U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<false>(idx0Buf, nkValueBuf, histogramsBuf, nkValueBuf, bottomK);
    HistogramsVFImpl<1U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true>(idx1Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
    HistogramsVFImpl<2U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true>(idx2Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
    HistogramsVFImpl<3U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true, true>(idx3Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0U, idx0Buf, idx1Buf, idx2Buf);
    const uint32_t outputTail = topK / kKeyLanes * kKeyLanes;
    const uint32_t paddedTopK = (topK + kKeyLanes - 1U) / kKeyLanes * kKeyLanes;
    AscendC::Duplicate(tmpIdxLocal[outputTail], uint32_t{0}, paddedTopK - outputTail);
    // 先输出大于阈值的编号。
    FindIdxOutputVFImpl<false>(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum);
    // 再按输入顺序补齐等于阈值的编号。
    FindIdxOutputVFImpl<true>(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum);
    if constexpr (ISOUTVALUE) {
        auto* outputValueBuf = reinterpret_cast<__ubuf__ uint32_t *>(outputValueLocal.GetPhyAddr());
        const uint16_t outputLoops = (topK + kKeyLanes - 1U) / kKeyLanes;
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, outputLoops);
    }
}

} // 命名空间结束。

namespace Detail {
// Whole-row high-prefix filtering preserves the original FP32 sortable keys.
__simd_callee__ inline void CompactHighPrefixPositions(__ubuf__ uint32_t *positions,
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *bucket0,
    __ubuf__ uint32_t *bucket1, __ubuf__ uint32_t *countOut, uint32_t aligned)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> high0, high1, threshold, zero, one, counts;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(high0, bucket0);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(high1, bucket1);
    Reg::ShiftLefts(high0, high0, static_cast<int16_t>(kHighByteShift), all);
    Reg::ShiftLefts(high1, high1, static_cast<int16_t>(kSecondByteShift), all);
    Reg::Or(threshold, high0, high1, all);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(one, 1U);
    Reg::Duplicate(counts, 0U);
    Reg::ClearSpr<SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore unalign;
    for (uint32_t offset = 0U; offset < aligned; offset += kKeyLanes) {
        Reg::RegTensor<uint32_t> key, ids, packed, increment;
        Reg::MaskReg selected, valid;
        Reg::LoadAlign(key, keys + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(ids), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::GE>(selected, key, threshold, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(valid, key, zero, all);
        Reg::And(selected, selected, valid, all);
        Reg::Select(increment, one, zero, selected);
        Reg::Add(counts, counts, increment, all);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, selected);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(positions, packed, unalign);
    }
    Reg::StoreUnAlignPost(positions, unalign);
    Reg::RegTensor<uint32_t> count;
    Reg::Reduce<Reg::ReduceType::SUM>(count, counts, all);
    uint32_t singleCount = 1U;
    auto single = Reg::UpdateMask<uint32_t>(singleCount);
    Reg::StoreAlign(countOut, count, single);
}

__simd_callee__ inline void GatherPrefixKeysInPlace(__ubuf__ uint32_t *keys,
    __ubuf__ uint32_t *positions, uint32_t count, uint32_t aligned)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero, limit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(limit, count);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < aligned; offset += kKeyLanes) {
        Reg::RegTensor<uint32_t> ids, lane, safe, value;
        Reg::MaskReg valid;
        Reg::LoadAlign(ids, positions + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lane), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, lane, limit, all);
        Reg::Select(safe, ids, zero, valid);
        Reg::Gather(value, keys, safe, all);
        Reg::Select(value, value, zero, valid);
        Reg::StoreAlign(keys + offset, value, all);
    }
}

// Count strict winners, then emit winners and the earliest threshold ties in ID order.
__simd_callee__ inline void GatherPrefixFinalIds(__ubuf__ uint32_t *outputIds,
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *positions,
    __ubuf__ uint32_t *thresholdPtr, uint32_t count, uint32_t keepCount)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> threshold, zero, one, seen, limit, greaterCounts;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(threshold, thresholdPtr);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(one, 1U);
    Reg::Duplicate(seen, 0U);
    Reg::Duplicate(limit, keepCount);
    Reg::Duplicate(greaterCounts, 0U);
    for (uint32_t offset = 0U; offset < count; offset += kKeyLanes) {
        uint32_t remaining = count - offset;
        auto active = Reg::UpdateMask<uint32_t>(remaining);
        Reg::RegTensor<uint32_t> key, increment;
        Reg::MaskReg greater;
        Reg::LoadAlign(key, keys + offset);
        Reg::Compare<uint32_t, CMPMODE::GT>(greater, key, threshold, active);
        Reg::Select(increment, one, zero, greater);
        Reg::Add(greaterCounts, greaterCounts, increment, all);
    }
    Reg::RegTensor<uint32_t> sum, broadcast;
    Reg::Reduce<Reg::ReduceType::SUM>(sum, greaterCounts, all);
    Reg::Gather(broadcast, sum, zero);
    Reg::Sub(limit, limit, broadcast, all);
    Reg::ClearSpr<SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore unalign;
    for (uint32_t offset = 0U; offset < count; offset += kKeyLanes) {
        uint32_t remaining = count - offset;
        auto active = Reg::UpdateMask<uint32_t>(remaining);
        Reg::RegTensor<uint32_t> key, id, packed, prefix, rank, increment;
        Reg::MaskReg greater, equal, withinLimit, takeEqual, selected;
        Reg::LoadAlign(key, keys + offset);
        Reg::LoadAlign(id, positions + offset);
        Reg::Compare<uint32_t, CMPMODE::GT>(greater, key, threshold, active);
        Reg::Compare<uint32_t, CMPMODE::EQ>(equal, key, threshold, active);
        Reg::Unsqueeze<uint32_t>(prefix, equal);
        Reg::Add(rank, prefix, seen, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(withinLimit, rank, limit, all);
        Reg::Select(increment, one, zero, equal);
        Reg::Reduce<Reg::ReduceType::SUM>(sum, increment, all);
        Reg::Gather(broadcast, sum, zero);
        Reg::Add(seen, seen, broadcast, all);
        Reg::And(takeEqual, equal, withinLimit, active);
        Reg::Or(selected, greater, takeEqual, active);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packed, id, selected);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIds, packed, unalign);
    }
    Reg::StoreUnAlignPost(outputIds, unalign);
}

// Keep radix stages inside two VFs, with one scalar compact-count decision between them.
__simd_callee__ inline void PrefixStageFence()
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__simd_vf__ void SelectHighPrefixVF(__ubuf__ uint32_t *input,
    __ubuf__ uint32_t *mapping, __ubuf__ uint32_t *histogram,
    __ubuf__ uint32_t *b0, __ubuf__ uint32_t *b1, __ubuf__ uint32_t *b2,
    __ubuf__ uint32_t *rank, __ubuf__ uint32_t *countOut, uint32_t aligned, uint32_t keepCount)
{
    PrefixStageFence();
    RadixDetail::HistogramsInline<0U>(histogram, input, b0, b1, b2, aligned / kHistogramInputElements);
    PrefixStageFence();
    RadixDetail::FindTargetBinInline<false>(b0, rank, histogram, rank, aligned - keepCount + 1U);
    PrefixStageFence();
    RadixDetail::HistogramsInline<1U>(histogram, input, b0, b1, b2, aligned / kHistogramInputElements);
    PrefixStageFence();
    RadixDetail::FindTargetBinInline<true>(b1, rank, histogram, rank, 0U);
    PrefixStageFence();
    CompactHighPrefixPositions(mapping, input, b0, b1, countOut, aligned);
}

__simd_vf__ void SelectLowSuffixVF(__ubuf__ uint32_t *input,
    __ubuf__ uint32_t *mapping, __ubuf__ uint32_t *histogram,
    __ubuf__ uint32_t *b0, __ubuf__ uint32_t *b1, __ubuf__ uint32_t *b2,
    __ubuf__ uint32_t *b3, __ubuf__ uint32_t *rank, __ubuf__ uint32_t *outputIds,
    uint32_t count, uint32_t aligned, uint32_t keepCount)
{
    GatherPrefixKeysInPlace(input, mapping, count, aligned);
    PrefixStageFence();
    RadixDetail::HistogramsInline<2U>(histogram, input, b0, b1, b2, aligned / kHistogramInputElements);
    PrefixStageFence();
    RadixDetail::FindTargetBinInline<true>(b2, rank, histogram, rank, 0U);
    PrefixStageFence();
    RadixDetail::HistogramsInline<3U>(histogram, input, b0, b1, b2, aligned / kHistogramInputElements);
    PrefixStageFence();
    RadixDetail::FindTargetBinInline<true, true>(b3, rank, histogram, rank, 0U, b0, b1, b2);
    PrefixStageFence();
    GatherPrefixFinalIds(outputIds, input, mapping, rank, count, keepCount);
}

} // namespace Detail

// Input keys order scores monotonically, with larger keys ranked first. Key 0
// excludes an element. Callers own score conversion and numerical validity.
// A mask policy further narrows `keep` using original zero-based element IDs.
// Set kEnabled=true and implement a __simd_callee__ static Apply method taking
// (Reg::MaskReg& keep, const Reg::RegTensor<uint32_t>& keys,
//  const Reg::RegTensor<uint32_t>& ids, const Reg::MaskReg& active,
//  __ubuf__ uint32_t* maskData, uint32_t row).
// maskData and row are caller-owned context, and keep initially equals active.
// NoMask compiles out the optional masking pass.
struct NoMask {
    static constexpr bool kEnabled = false;
};

enum class SelectionStatus : uint8_t {
    Success,
    Unsupported,
    InsufficientCandidates,
};

namespace Detail {
template <class MaskPolicy>
__simd_vf__ void ApplyMaskVF(__ubuf__ uint32_t *keys, uint32_t count,
    uint32_t aligned, __ubuf__ uint32_t *maskData, uint32_t row)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::Duplicate(zero, 0U);
    for (uint32_t offset = 0U; offset < aligned; offset += kKeyLanes) {
        uint32_t remaining = count > offset ? count - offset : 0U;
        auto active = Reg::UpdateMask<uint32_t>(remaining);
        Reg::RegTensor<uint32_t> key, ids;
        Reg::LoadAlign(key, keys + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(ids), static_cast<int32_t>(offset));
        auto keep = active;
        MaskPolicy::Apply(keep, key, ids, active, maskData, row);
        Reg::And(keep, keep, active, all);
        Reg::Select(key, key, zero, keep);
        Reg::StoreAlign(keys + offset, key, all);
    }
}
} // namespace Detail

// Place input keys at the start of workspace and zero-pad to kHistogramInputElements.
// workspace must satisfy kStoreAlignmentBytes and contain Workspace::kWorkspaceBytes bytes.
// outputIds must hold keepCount uint32 entries, satisfy kStoreAlignmentBytes, and must not
// overlap the first Workspace::kWorkspaceBytes bytes of workspace. Success emits ascending IDs,
// selecting smaller IDs at ties.
// The caller synchronizes V to MTE3 before copying outputIds to GM, or V to S
// before reading it with Scalar. SelectHighPrefixIds does not own GM loads, stores or TPipe.
// InsufficientCandidates leaves outputIds untouched. Input workspace is mutable
// on all paths, so a caller taking a fallback must reload or reconstruct keys.
// The Workspace type sets row and keep capacities; count>=keepCount is required.
template <class MaskPolicy = NoMask, class Workspace = DefaultHighPrefixWorkspace>
__aicore__ inline SelectionStatus SelectHighPrefixIds(LocalTensor<uint8_t> workspace,
    LocalTensor<uint32_t> outputIds, uint32_t count, uint32_t keepCount,
    __ubuf__ uint32_t *maskData = nullptr, uint32_t row = 0U)
{
    if (count > Workspace::kMaxElements || keepCount == 0U || keepCount > Workspace::kMaxKeepCount) {
        return SelectionStatus::Unsupported;
    }
    if (count < keepCount) {
        return SelectionStatus::InsufficientCandidates;
    }
    auto keys = workspace.ReinterpretCast<uint32_t>();
    auto positions = workspace[Workspace::kPositionsOffset].template ReinterpretCast<uint32_t>();
    auto hist = workspace[Workspace::kHistogramOffset].template ReinterpretCast<uint32_t>();
    auto bucket0 = hist[Workspace::kBucket0Offset];
    auto bucket1 = hist[Workspace::kBucket1Offset];
    auto bucket2 = hist[Workspace::kBucket2Offset];
    auto bucket3 = hist[Workspace::kBucket3Offset];
    auto kth = hist[Workspace::kRankOffset];
    auto countOut = hist[Workspace::kCountOffset];
    const uint32_t aligned = (count + kHistogramInputElements - 1U) / kHistogramInputElements * kHistogramInputElements;
    auto input = reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr());
    auto mapping = reinterpret_cast<__ubuf__ uint32_t *>(positions.GetPhyAddr());
    auto histogram = reinterpret_cast<__ubuf__ uint32_t *>(hist.GetPhyAddr());
    auto b0 = reinterpret_cast<__ubuf__ uint32_t *>(bucket0.GetPhyAddr());
    auto b1 = reinterpret_cast<__ubuf__ uint32_t *>(bucket1.GetPhyAddr());
    auto b2 = reinterpret_cast<__ubuf__ uint32_t *>(bucket2.GetPhyAddr());
    auto b3 = reinterpret_cast<__ubuf__ uint32_t *>(bucket3.GetPhyAddr());
    auto rank = reinterpret_cast<__ubuf__ uint32_t *>(kth.GetPhyAddr());
    if constexpr (MaskPolicy::kEnabled) {
        Detail::ApplyMaskVF<MaskPolicy>(input, count, aligned, maskData, row);
    }
    Detail::SelectHighPrefixVF(input, mapping, histogram, b0, b1, b2, rank,
        reinterpret_cast<__ubuf__ uint32_t *>(countOut.GetPhyAddr()), aligned, keepCount);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t compactCount = countOut.GetValue(0U);
    if (compactCount < keepCount) {
        return SelectionStatus::InsufficientCandidates;
    }
    const uint32_t compactAligned = (compactCount + kHistogramInputElements - 1U) / kHistogramInputElements * kHistogramInputElements;
    Detail::SelectLowSuffixVF(input, mapping, histogram, b0, b1, b2, b3, rank,
        reinterpret_cast<__ubuf__ uint32_t *>(outputIds.GetPhyAddr()),
        compactCount, compactAligned, keepCount);
    return SelectionStatus::Success;
}

// Direct four-byte radix selection for one caller-prepared window. Input keys
// rank larger values first; validLen is a multiple of kHistogramInputElements and topK<=validLen.
// Output positions are ordered by strict winners followed by threshold ties;
// the caller owns final ID/value ordering, masks, padding and scratch capacity.
// Each bucket holds kHistogramBins uint32 entries; nextRank holds kKeyLanes entries.
// There is no row-length algorithm selection in this entry.
template <bool ReturnValues = false>
__aicore__ inline void SelectRadixWindow(const LocalTensor<uint32_t> &positions,
    const LocalTensor<uint32_t> &outputValues, const LocalTensor<uint32_t> &keys,
    const LocalTensor<uint32_t> &histogram, const LocalTensor<uint32_t> &bucket0,
    const LocalTensor<uint32_t> &bucket1, const LocalTensor<uint32_t> &bucket2,
    const LocalTensor<uint32_t> &bucket3, const LocalTensor<uint32_t> &nextRank,
    uint32_t topK, uint32_t validLen)
{
    RadixDetail::LiTopKVF<ReturnValues>(positions, outputValues, keys, histogram,
        bucket0, bucket1, bucket2, bucket3, nextRank, topK, validLen);
}

} // namespace SparseIndexer::Common::Primitive::TopK
#endif
