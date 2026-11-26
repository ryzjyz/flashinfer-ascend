/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_LINEAR_ARENA_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_LINEAR_ARENA_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "contracts/common.hpp"

namespace SparseIndexer::Common {

/**
 * Typed linear allocator for a Catlass UB/L1 arena. Allocation happens only
 * during policy construction; the hot compute loop receives plain tensors.
 */
template <class Buffer_>
class LinearArena {
public:
    using Buffer = Buffer_;

    CATLASS_DEVICE explicit LinearArena(Buffer& buffer, uint32_t baseOffset = 0U)
        : buffer_(buffer), offset_(baseOffset)
    {
    }

    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> Take(uint32_t count, uint32_t alignment = 1U)
    {
        offset_ = AlignUp(offset_, alignment);
        auto tensor = buffer_.template GetBufferByByte<T>(offset_);
        offset_ += count * static_cast<uint32_t>(sizeof(T));
        return tensor;
    }

    CATLASS_DEVICE uint32_t UsedBytes() const { return offset_; }

private:
    Buffer& buffer_;
    uint32_t offset_{0};
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_LINEAR_ARENA_HPP
