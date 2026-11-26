/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_SCORE_BUFFER_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_SCORE_BUFFER_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"

namespace SparseIndexer::Common {

struct ScoreBufferGeometry {
    uint32_t workRows{0};
    uint32_t heads{0};
    uint32_t colsPerAiv{0};
};

CATLASS_HOST_DEVICE constexpr uint32_t ScoreElementsPerAiv(ScoreBufferGeometry const& geometry)
{
    return geometry.workRows * geometry.heads * geometry.colsPerAiv;
}

CATLASS_HOST_DEVICE constexpr uint32_t ScoreBufferCount(
    uint32_t scoreElements, uint64_t ubSize, uint64_t scratchReserve = 64U * 1024U)
{
    const uint64_t twoScores = 2ULL * scoreElements * sizeof(float);
    return twoScores + scratchReserve <= ubSize ? 2U : 1U;
}

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_SCORE_BUFFER_HPP
