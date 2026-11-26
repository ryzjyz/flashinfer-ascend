/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_PROBLEM_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_PROBLEM_HPP

#include <cstdint>

namespace SparseIndexer::Common {

enum class IndexerCacheKind : uint8_t { Continuous = 0, Paged = 1 };
enum class IndexerMaskKind : uint8_t { None = 0, CompressedMask3 = 3, RegionCausal = 4 };
enum class IndexerOutputKind : uint8_t { IndicesOnly = 0, ScoreAndIndices = 1 };
enum class IndexerSelectionOrder : uint8_t { QliRawSlot = 0, ScoreDescending = 1, IndexAscending = 2 };

struct IndexerProblem {
    uint32_t batch{1};
    uint32_t totalQ{0};
    uint32_t groups{1};
    uint32_t heads{0};
    uint32_t headDim{0};
    uint64_t totalK{0};
    IndexerCacheKind cacheKind{IndexerCacheKind::Paged};
    IndexerMaskKind maskKind{IndexerMaskKind::CompressedMask3};
    IndexerOutputKind outputKind{IndexerOutputKind::IndicesOnly};
    IndexerSelectionOrder selectionOrder{IndexerSelectionOrder::QliRawSlot};
};

struct RequestView {
    uint32_t requestId{0};
    uint32_t qBegin{0};
    uint32_t qCount{0};
    uint32_t kBegin{0};
    uint32_t kCount{0};
    uint32_t visibleK{0};
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_PROBLEM_HPP
