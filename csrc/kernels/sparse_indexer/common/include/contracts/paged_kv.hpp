/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_PAGED_KV_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_PAGED_KV_HPP

#include <cstdint>

namespace SparseIndexer::Common {

/// Page-addressing geometry shared by key and scale loaders.
struct PagedKvGeometry {
    uint32_t d{0};
    uint32_t blockSize{0};
    uint32_t maxBlocksPerBatch{0};
    uint32_t numPages{0};
    uint64_t keyStride0{0};
    uint64_t scaleStride0{0};
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_PAGED_KV_HPP
