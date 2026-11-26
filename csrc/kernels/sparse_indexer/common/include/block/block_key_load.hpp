/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_KEY_LOAD_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_KEY_LOAD_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/matrix_coord.hpp"

#include "contracts/common.hpp"
#include "contracts/paged_kv.hpp"

namespace SparseIndexer::Common::Block {

/// Address data resolved by BlockSchedule for one request-local K tile.
struct KeyCoord {
    uint32_t globalNStart{0};
    uint32_t requestNStart{0};
    uint32_t batchIdx{0};
};

/**
 * Contiguous Key → L1B (default path).
 * Load uses logical (kPart, nStart) into ColumnMajor GM [D, S2].
 */
template <class ArchTag_, class BType_>
class ContiguousKeyLoader {
public:
    struct Params {};

    static constexpr bool kPaged = false;
    using ArchTag = ArchTag_;
    using BType = BType_;
    using DtypeB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;

    CATLASS_DEVICE
    ContiguousKeyLoader() = default;

    CATLASS_DEVICE
    void Setup(
        AscendC::GlobalTensor<DtypeB> const& gmB, LayoutB const& layoutB,
        Params const& /*params*/,
        AscendC::GlobalTensor<int32_t> const& /*gmTable*/,
        AscendC::GlobalTensor<DtypeB> const& /*gmZero*/)
    {
        gmB_ = gmB;
        layoutB_ = layoutB;
    }

    template <class TensorL1, class LayoutL1>
    CATLASS_DEVICE void Load(
        TensorL1 const& l1B, LayoutL1 const& layoutL1, uint32_t kPartStart, uint32_t actualK,
        KeyCoord const& keyCoord, uint32_t actualN)
    {
        int64_t offsetB = layoutB_.GetOffset(Catlass::MatrixCoord(kPartStart, keyCoord.globalNStart));
        copyGmToL1B_(
            l1B, gmB_[offsetB], layoutL1, layoutB_.GetTileLayout(Catlass::MatrixCoord(actualK, actualN)));
    }

private:
    AscendC::GlobalTensor<DtypeB> gmB_;
    LayoutB layoutB_{};
    CopyGmToL1B copyGmToL1B_;
};

/**
 * PA_BBND Key → L1B via block_table (M4.1 / D3).
 *
 * Geometry:
 * - Pages GM is [numPages, blockSize, D] (row-major); logical n is request-local
 *   via ``nStart - cuSeqlensK[batchIdx]``.
 * - Each Load splits ``actualN`` into page-aligned runs
 *   ``[offInPage, blockSize - offInPage]`` so a single s2Tile may cross pages.
 * - Illegal / OOB page ids (``<0`` or ``>= numPages``) copy from a host-zeroed
 *   page buffer (ColumnMajor(d, blockSize) view starting at token 0).
 *
 * Note: production gather quality (batched page copy / less zeroing) is optional
 * M5 work; this loader prioritizes correct semantics over peak PA bandwidth.
 */
template <class ArchTag_, class BType_>
class PagedKeyLoader {
public:
    struct Params {
        uint32_t d{0};
        uint32_t blockSize{0};
        uint32_t maxBlocksPerBatch{0};
        uint32_t numPages{0};
        uint64_t keyStride0{0};
    };

    static constexpr bool kPaged = true;
    using ArchTag = ArchTag_;
    using BType = BType_;
    using DtypeB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;

    CATLASS_DEVICE
    PagedKeyLoader() = default;

    CATLASS_DEVICE
    void Setup(
        AscendC::GlobalTensor<DtypeB> const& gmPages, LayoutB const& /*layoutB*/,
        Params const& params,
        AscendC::GlobalTensor<int32_t> const& gmTable,
        AscendC::GlobalTensor<DtypeB> const& gmZero)
    {
        gmPages_ = gmPages;
        blockSize_ = params.blockSize;
        maxBlocks_ = params.maxBlocksPerBatch;
        numPages_ = params.numPages;
        d_ = params.d;
        keyStride0_ = params.keyStride0 != 0
            ? params.keyStride0
            : static_cast<uint64_t>(blockSize_) * d_;
        gmTable_ = gmTable;
        gmZero_ = gmZero;
    }

    template <class TensorL1, class LayoutL1>
    CATLASS_DEVICE void Load(
        TensorL1 const& l1B, LayoutL1 const& layoutL1, uint32_t kPartStart, uint32_t actualK,
        KeyCoord const& keyCoord, uint32_t actualN)
    {
        const uint32_t localN = keyCoord.requestNStart;

        uint32_t copyCol = 0;
        while (copyCol < actualN) {
            uint32_t tok = localN + copyCol;
            uint32_t logicalBlock = (blockSize_ == 0) ? 0U : (tok / blockSize_);
            uint32_t offInPage = (blockSize_ == 0) ? 0U : (tok % blockSize_);
            uint32_t run = actualN - copyCol;
            if (blockSize_ > offInPage) {
                run = AscendC::Std::min(run, blockSize_ - offInPage);
            }

            int32_t pageId = -1;
            if (logicalBlock < maxBlocks_) {
                pageId = gmTable_.GetValue(static_cast<int64_t>(keyCoord.batchIdx) * maxBlocks_ + logicalBlock);
            }

            AscendC::GlobalTensor<DtypeB> src;
            if (pageId >= 0 && static_cast<uint32_t>(pageId) < numPages_) {
                int64_t srcOff = static_cast<int64_t>(pageId) * static_cast<int64_t>(keyStride0_) +
                    static_cast<int64_t>(offInPage) * d_ + kPartStart;
                src = gmPages_[srcOff];
            } else {
                // Illegal page → zeros (zero page is ColumnMajor(d, blockSize) starting at 0).
                src = gmZero_[static_cast<int64_t>(kPartStart)];
            }

            int64_t dstOff = layoutL1.GetOffset(Catlass::MatrixCoord(0, copyCol));
            auto layoutSrc = LayoutB::template MakeLayout<DtypeB>(d_, run);
            copyGmToL1B_(
                l1B[dstOff], src, layoutL1, layoutSrc.GetTileLayout(Catlass::MatrixCoord(actualK, run)));
            copyCol += run;
        }
    }

private:
    AscendC::GlobalTensor<DtypeB> gmPages_;
    AscendC::GlobalTensor<int32_t> gmTable_;
    AscendC::GlobalTensor<DtypeB> gmZero_;
    CopyGmToL1B copyGmToL1B_;
    uint32_t blockSize_{0};
    uint32_t maxBlocks_{0};
    uint32_t numPages_{0};
    uint32_t d_{0};
    uint64_t keyStride0_{0};
};

/// Compile-time cache-side policy used by the QK MMAD block.
template <class Dtype_, bool Paged_>
struct BlockKeyLoad {
    using Dtype = Dtype_;
    using BType = Catlass::Gemm::GemmType<Dtype, Catlass::layout::ColumnMajor>;
    using Loader = std::conditional_t<Paged_,
        PagedKeyLoader<Catlass::Arch::Ascend950, BType>,
        ContiguousKeyLoader<Catlass::Arch::Ascend950, BType>>;
    static constexpr bool kPaged = Paged_;
};

/**
 * Private L1B ring for one BlockQkMmad. It owns only the key loader and the
 * three key-buffer events; the Q/L0/L0C lifecycle remains in BlockQkMmad.
 */
template <class DtypeB_, class KeyLoader_>
class KeyRing {
public:
    using DtypeB = DtypeB_;
    using KeyLoader = KeyLoader_;
    using LayoutB = typename KeyLoader::LayoutB;
    using LoaderParams = typename KeyLoader::Params;
    using TensorB = AscendC::LocalTensor<DtypeB>;

    CATLASS_DEVICE KeyRing() = default;

    CATLASS_DEVICE void Setup(
        TensorB const& l1B, AscendC::GlobalTensor<DtypeB> const& gmB, LayoutB const& layoutB,
        uint32_t problemK, uint32_t kPart, uint32_t keyPartElems, uint32_t keyBufElems,
        LoaderParams const& params, AscendC::GlobalTensor<int32_t> const& gmTable,
        AscendC::GlobalTensor<DtypeB> const& gmZero)
    {
        l1B_ = l1B;
        problemK_ = problemK;
        kPart_ = kPart;
        keyPartElems_ = keyPartElems;
        keyBufElems_ = keyBufElems;
        keyLoader_.Setup(gmB, layoutB, params, gmTable, gmZero);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        initialized_ = true;
    }

    CATLASS_DEVICE ~KeyRing()
    {
        if (!initialized_) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    CATLASS_DEVICE static constexpr uint32_t Event(uint32_t bufferIdx)
    {
        return bufferIdx == 0U ? EVENT_ID0 : (bufferIdx == 1U ? EVENT_ID2 : EVENT_ID3);
    }

    CATLASS_DEVICE TensorB Buffer(uint32_t bufferIdx, uint32_t kIdx) const
    {
        return l1B_[bufferIdx * keyBufElems_ + kIdx * keyPartElems_];
    }

    CATLASS_DEVICE void WaitReady(uint32_t bufferIdx) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Event(bufferIdx));
    }

    CATLASS_DEVICE void Release(uint32_t bufferIdx) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Event(bufferIdx));
    }

    CATLASS_DEVICE void Prefetch(
        uint32_t bufferIdx, KeyCoord const& keyCoord, uint32_t actualN, uint32_t alignedN)
    {
        const uint32_t kTileCount = SparseIndexer::Common::CeilDiv(problemK_, kPart_);
        auto layoutBInL1 = Catlass::layout::nZ::template MakeLayout<DtypeB>(kPart_, alignedN);
        const uint32_t eventId = Event(bufferIdx);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        for (uint32_t kIdx = 0; kIdx < kTileCount; ++kIdx) {
            const uint32_t kStart = kIdx * kPart_;
            const uint32_t actualK = AscendC::Std::min(kPart_, problemK_ - kStart);
            keyLoader_.Load(Buffer(bufferIdx, kIdx), layoutBInL1, kStart, actualK, keyCoord, actualN);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

private:
    TensorB l1B_;
    KeyLoader keyLoader_;
    uint32_t problemK_{0};
    uint32_t kPart_{0};
    uint32_t keyPartElems_{0};
    uint32_t keyBufElems_{0};
    bool initialized_{false};
};

} // namespace SparseIndexer::Common::Block


#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_KEY_LOAD_HPP
