/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP

#include <cstdint>

#include "kernel_operator.h"
#include "catlass/detail/macros.hpp"

namespace SparseIndexer::Common {

template <class DtypeScore>
class NoMaskPolicy {
public:
    CATLASS_DEVICE static uint32_t VisibleLen(
        uint32_t, uint32_t, uint32_t kAct, uint32_t, int32_t)
    {
        return kAct;
    }

    CATLASS_DEVICE
    NoMaskPolicy(
        AscendC::LocalTensor<DtypeScore> const&, AscendC::LocalTensor<DtypeScore> const&,
        AscendC::LocalTensor<DtypeScore> const&)
    {}

    CATLASS_DEVICE
    void Apply(
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t,
        bool, int32_t, int32_t)
    {}
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP
