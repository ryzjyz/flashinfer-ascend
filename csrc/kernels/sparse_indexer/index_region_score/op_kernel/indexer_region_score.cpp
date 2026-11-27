// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。
// 按 TilingKey 分支解析对应的缓存 ABI。
#include "kernel_operator.h"
#include "arch35/indexer_region_score_pipeline.h"
#include "arch35/block/block_topk.hpp"
#include "arch35/weighted_pipeline.hpp"

using IndexerRegionScore::RegionScoreTilingData;

using namespace IndexerRegionScorePaged;

namespace IndexerRegionScorePaged {
using ContinuousKeyCacheStage = IndexerRegionScore::Block::StageType<
    IndexerRegionScore::Block::Stage::KeyCache,
    IndexerRegionScore::Block::KeyCachePolicy<IndexerRegionScore::Block::KeyCacheKind::Continuous>>;

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((always_inline)) inline void RunPagedInline(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // 在被选中的实现内部解析完整 POD 一次。
    PagedTilingData sharedTiling{};
    uint32_t *local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunPagedCubeWithStrategies(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // 在被选中的实现内部解析完整 POD 一次。
    PagedTilingData sharedTiling{};
    uint32_t *local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
}

template <typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunD128PagedCubeIsolated(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    PagedTilingData sharedTiling{};
    auto local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    if (sharedTiling.pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
        D128QueryResidentRunner<PartialPagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
    } else {
        D128QueryResidentRunner<PagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
    }
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ inline void RunGridWithStrategies(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
    GM_ADDR actualKvLengths, GM_ADDR pagedKeySummary, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // Cube 入口读取分页搬运模式；被选中实现解析完整 ABI 一次。
    if constexpr (CacheStrategy::IS_PAGED) {
        if constexpr (sizeof(QkInputType) == 2U && MaskStrategy::CAUSAL) {
            const uint32_t proxyDim = reinterpret_cast<__gm__ uint32_t *>(tiling)[
                offsetof(PagedTilingData, proxyDim) / sizeof(uint32_t)];
            if (proxyDim == 128U) {
                if ASCEND_IS_AIC {
                    RunD128PagedCubeIsolated<MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(query, weights,
                        pagedKeySummary, actualKvLengths, blockTable, cuSeqLens, output,
                        selectedRegionIds, workspace, tiling, pipe);
                } else {
                PagedTilingData sharedTiling{};
                auto local = reinterpret_cast<uint32_t *>(&sharedTiling);
                auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
                for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
                    local[i] = global[i];
                }
                if (sharedTiling.pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
                    D128QueryResidentRunner<PartialPagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
                    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
                } else {
                    D128QueryResidentRunner<PagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
                    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
                }
                }
                return;
            }
        }
        if ASCEND_IS_AIC {
            auto global = reinterpret_cast<__gm__ uint64_t *>(tiling);
            const uint64_t pageLoadMode = global[offsetof(PagedTilingData, pageLoadMode) / sizeof(uint64_t)];
            if (pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
                RunPagedCubeWithStrategies<PartialPagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
                    query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                    output, selectedRegionIds, workspace, tiling, pipe);
            } else {
                // 页对齐路径保持内联，避免短 case 支付额外函数调用成本。
                RunPagedInline<PagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
                    query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                    output, selectedRegionIds, workspace, tiling, pipe);
            }
        } else {
            RunPagedInline<PagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
                query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                output, selectedRegionIds, workspace, tiling, pipe);
        }
        return;
    } else {
        GET_TILING_DATA_WITH_STRUCT(RegionScoreTilingData, tilingData, tiling);
        PagedTilingData sharedTiling = NormalizeContinuousTiling(tilingData);
        RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, tilingData.lengthStrideG, pipe);
    }
}
} // 命名空间 IndexerRegionScorePaged

namespace IndexerRegionScoreWeighted {
using ContinuousKeyCacheStage = IndexerRegionScore::Block::StageType<
    IndexerRegionScore::Block::Stage::KeyCache,
    IndexerRegionScore::Block::KeyCachePolicy<IndexerRegionScore::Block::KeyCacheKind::Continuous>>;

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((always_inline)) inline void RunPagedInline(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // 在被选中的实现内部解析完整 POD 一次。
    PagedTilingData sharedTiling{};
    uint32_t *local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunPagedCubeWithStrategies(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // 在被选中的实现内部解析完整 POD 一次。
    PagedTilingData sharedTiling{};
    uint32_t *local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
}

template <typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunD128PagedCubeIsolated(GM_ADDR query, GM_ADDR weights,
    GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    PagedTilingData sharedTiling{};
    auto local = reinterpret_cast<uint32_t *>(&sharedTiling);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
    if (sharedTiling.pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
        D128QueryResidentRunner<PartialPagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
    } else {
        D128QueryResidentRunner<PagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
    }
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunCubeWithStrategies(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
    GM_ADDR actualKvLengths, GM_ADDR pagedKeySummary, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // Cube 入口读取分页搬运模式；被选中实现解析完整 ABI 一次。
    if constexpr (CacheStrategy::IS_PAGED) {
        if constexpr (sizeof(QkInputType) == 2U && MaskStrategy::CAUSAL) {
            const uint32_t proxyDim = reinterpret_cast<__gm__ uint32_t *>(tiling)[
                offsetof(PagedTilingData, proxyDim) / sizeof(uint32_t)];
            if (proxyDim == 128U) {
                RunD128PagedCubeIsolated<MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(query, weights,
                    pagedKeySummary, actualKvLengths, blockTable, cuSeqLens, output,
                    selectedRegionIds, workspace, tiling, pipe);
                return;
            }
        }

        auto global = reinterpret_cast<__gm__ uint64_t *>(tiling);
        const uint64_t pageLoadMode = global[offsetof(PagedTilingData, pageLoadMode) / sizeof(uint64_t)];
        if (pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
            RunPagedCubeWithStrategies<PartialPagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
                query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                output, selectedRegionIds, workspace, tiling, pipe);
        } else {
            // 页对齐路径保持内联，避免短 case 支付额外函数调用成本。
            RunPagedInline<PagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
                query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                output, selectedRegionIds, workspace, tiling, pipe);
        }

        return;
    } else {
        GET_TILING_DATA_WITH_STRUCT(RegionScoreTilingData, tilingData, tiling);
        PagedTilingData sharedTiling = NormalizeContinuousTiling(tilingData);
        RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, tilingData.lengthStrideG, pipe);
    }
}
template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ __attribute__((noinline)) void RunVectorWithStrategies(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
    GM_ADDR actualKvLengths, GM_ADDR pagedKeySummary, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    // Vector 按请求范围处理归约与选择；D128 使用独立驻留参数。
    if constexpr (CacheStrategy::IS_PAGED) {
        if constexpr (sizeof(QkInputType) == 2U && MaskStrategy::CAUSAL) {
            const uint32_t proxyDim = reinterpret_cast<__gm__ uint32_t *>(tiling)[
                offsetof(PagedTilingData, proxyDim) / sizeof(uint32_t)];
            if (proxyDim == 128U) {
                PagedTilingData sharedTiling{};
                auto local = reinterpret_cast<uint32_t *>(&sharedTiling);
                auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
                for (uint32_t i = 0U; i < sizeof(PagedTilingData) / sizeof(uint32_t); ++i) {
                    local[i] = global[i];
                }
                if (sharedTiling.pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
                    D128QueryResidentRunner<PartialPagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
                    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
                } else {
                    D128QueryResidentRunner<PagedD128KeyCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL> runner;
                    runner.Run(query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
                        output, selectedRegionIds, workspace, sharedTiling, sharedTiling.reservedStride0, pipe);
                }

                return;
            }
        }

        RunPagedInline<PagedCacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, pagedKeySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, tiling, pipe);
        return;
    } else {
        GET_TILING_DATA_WITH_STRUCT(RegionScoreTilingData, tilingData, tiling);
        PagedTilingData sharedTiling = NormalizeContinuousTiling(tilingData);
        RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, false, ALLOW_H4_TAIL> runner;
        runner.Run(query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens,
            output, selectedRegionIds, workspace, sharedTiling, tilingData.lengthStrideG, pipe);
    }
}
} // 命名空间 IndexerRegionScoreWeighted

#include "arch35/dispatch.hpp"

extern "C" __global__ __aicore__ void fi_indexer_region_score(
    GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary, GM_ADDR actualKvLengths,
    GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling)
{
    // Both cache strategies consume the single required Key input.
    GM_ADDR pagedKeySummary = keySummary;
    // GE sizes its raw tiling buffer from the registered maximum POD. The
    // continuous branch still explicitly decodes RegionScoreTilingData.
    static_assert(sizeof(PagedTilingData) >= sizeof(RegionScoreTilingData),
                  "default tiling capacity must cover both cache layouts");
    REGISTER_TILING_DEFAULT(PagedTilingData);
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK3_H4_TAIL_TOPK)) {
        RunWithStrategies<ContinuousKeyCacheStage, RightDownCausalMaskStrategy, FusedTopKStrategy, true>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
        return;
    }
    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK3_H4_TAIL_TOPK)) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
        auto data = reinterpret_cast<__gm__ uint32_t *>(tiling);
        const uint32_t proxyDim = data[offsetof(PagedTilingData, proxyDim) / sizeof(uint32_t)];
        const uint32_t selection = data[offsetof(PagedTilingData, selectionConfig) / sizeof(uint32_t)];
        if (proxyDim == 256U && (selection & IndexerRegionScore::Block::kSelectionKeepMask) ==
            IndexerRegionScore::FusedSelect::kNativeTopK) {
            RunWithStrategies<PagedCacheStrategy, RightDownCausalMaskStrategy, Native512TopKStrategy, true>(
                query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
                cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
            return;
        }
#endif
        RunWithStrategies<PagedCacheStrategy, RightDownCausalMaskStrategy, FusedTopKStrategy, true>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
        return;
    }

    // 使用数值宏让预编译器识别所有特化。
    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK0_TOPK)) {
        RunWithStrategies<ContinuousKeyCacheStage, NoMaskStrategy, FusedTopKStrategy>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
        return;
    }
    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK0_TOPK)) {
        RunWithStrategies<PagedCacheStrategy, NoMaskStrategy, FusedTopKStrategy>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
        return;
    }
    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK3_TOPK)) {
        RunWithStrategies<ContinuousKeyCacheStage, RightDownCausalMaskStrategy, FusedTopKStrategy>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
        return;
    }
    if (TILING_KEY_IS(INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK3_TOPK)) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
        if constexpr (sizeof(QkInputType) == 2U) {
            auto data = reinterpret_cast<__gm__ uint32_t *>(tiling);
            const uint32_t proxyDim = data[offsetof(PagedTilingData, proxyDim) / sizeof(uint32_t)];
            const uint32_t selection = data[offsetof(PagedTilingData, selectionConfig) / sizeof(uint32_t)];
            if (proxyDim == 256U && (selection & IndexerRegionScore::Block::kSelectionKeepMask) ==
            IndexerRegionScore::FusedSelect::kNativeTopK) {
                RunWithStrategies<PagedCacheStrategy, RightDownCausalMaskStrategy, Native512TopKStrategy>(
                    query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
                    cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
                return;
            }
        }
#endif
        RunWithStrategies<PagedCacheStrategy, RightDownCausalMaskStrategy, FusedTopKStrategy>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, &pipe);
    }
}
