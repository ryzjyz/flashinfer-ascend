// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。
#ifndef INDEXER_REGION_SCORE_TILING_DATA_H
#define INDEXER_REGION_SCORE_TILING_DATA_H

#include <cstddef>
#include <cstdint>

// Kernel 预编译需通过宏解析 TILING_KEY_IS，以下宏枚举 MakeTilingKey 组合。
#define INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK0_TOPK 0
#define INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK0_TOPK 1
#define INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK3_TOPK 12
#define INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK3_TOPK 13
#define INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK3_H4_TAIL_TOPK 140ULL
#define INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK3_H4_TAIL_TOPK 141ULL

namespace IndexerRegionScore {
constexpr uint32_t SCORE_PLAN_LINE_BYTES = 128U;

#if defined(__CCE_AICORE__)
#define IRS_SCORE_PLAN_INLINE __aicore__ inline constexpr
#else
#define IRS_SCORE_PLAN_INLINE inline constexpr
#endif

IRS_SCORE_PLAN_INLINE uint32_t LimitScorePlanSlots(uint32_t epochs, uint32_t pairs)
{
    return epochs < pairs ? epochs : pairs;
}

// Bounded execution retains one immutable cut slot per epoch in a core-sized window.
IRS_SCORE_PLAN_INLINE uint32_t ScorePlanWindowSlots(uint32_t totalTokens, uint32_t batchTokens, uint32_t pairs)
{
    return LimitScorePlanSlots(batchTokens == 0U ? 1U :
        totalTokens / batchTokens + (totalTokens % batchTokens != 0U), pairs);
}

IRS_SCORE_PLAN_INLINE uint64_t ScorePlanWorkspaceBytes(uint32_t requests, uint32_t pairs,
    uint32_t totalTokens = 0U, uint32_t batchTokens = 0U)
{
    return requests <= 1U || pairs <= 1U ? 0U :
        (static_cast<uint64_t>(pairs) + 2U) * SCORE_PLAN_LINE_BYTES *
            ScorePlanWindowSlots(totalTokens, batchTokens, pairs) +
        static_cast<uint64_t>(pairs) * SCORE_PLAN_LINE_BYTES +
        (static_cast<uint64_t>(requests) * 8U * sizeof(uint32_t) + SCORE_PLAN_LINE_BYTES - 1U) /
            SCORE_PLAN_LINE_BYTES * SCORE_PLAN_LINE_BYTES;
}

#undef IRS_SCORE_PLAN_INLINE

// Preserve the region-ID order bit in the existing selectionConfig POD layout.
inline constexpr uint32_t kSelectionRegionIdOrder = 1U << 16U;

// 位 [1:0] 为缓存模式：0 连续、1 分页，2/3 保留。
// 位 [5:2] 为 maskMode 原值，预留 0..9，目前支持 0/3。
// 位 [6] 保留。
// 位 [7] 启用逻辑 R 尾块下完整 H4 tile 的批量归约。
// TopK 始终执行；位 [63:8] 必须为零。
// TopK 数量、排序/对齐标志和权重类型仍保存在 POD 中。
constexpr uint32_t TILING_KEY_CACHE_SHIFT = 0U;
constexpr uint32_t TILING_KEY_CACHE_WIDTH = 2U;
constexpr uint32_t TILING_KEY_MASK_SHIFT = TILING_KEY_CACHE_SHIFT + TILING_KEY_CACHE_WIDTH;
constexpr uint64_t TILING_KEY_H4_TAIL = 1ULL << 7U;
// 调用方编码前校验模式；不得截断保留值或将 maskMode 折叠为布尔值，以免误选其他策略。
inline constexpr uint64_t MakeTilingKey(uint32_t cacheMode, uint32_t maskMode)
{
    return (static_cast<uint64_t>(cacheMode) << TILING_KEY_CACHE_SHIFT) |
           (static_cast<uint64_t>(maskMode) << TILING_KEY_MASK_SHIFT);
}

static_assert(MakeTilingKey(0U, 0U) == INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK0_TOPK &&
              MakeTilingKey(1U, 0U) == INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK0_TOPK &&
              MakeTilingKey(0U, 3U) == INDEXER_REGION_SCORE_TILING_KEY_CONTINUOUS_MASK3_TOPK &&
              MakeTilingKey(1U, 3U) == INDEXER_REGION_SCORE_TILING_KEY_PAGED_MASK3_TOPK,
              "tiling-key macros must match the shared encoder");

// 因果计划协议有固定开销；Host 工作区计算与设备执行共用此门限以保持 mailbox 决策一致。
// 非跳过结果环在 48 个完整 NR=128 tile 内稳定；从第 49 个起启用流式因果计划和后缀跳过。
constexpr uint64_t CAUSAL_SKIP_MIN_SEQ_K = 6145U;
constexpr uint32_t CAUSAL_MAILBOX_MAX_CUBE_CORES = 36U;
constexpr uint32_t CAUSAL_MAILBOX_SLOTS = 2U;
constexpr uint32_t CAUSAL_MAILBOX_SLOT_BYTES = 32U;
constexpr uint32_t CAUSAL_MAILBOX_BYTES = 4096U;

static_assert(CAUSAL_MAILBOX_MAX_CUBE_CORES * CAUSAL_MAILBOX_SLOTS *
                  CAUSAL_MAILBOX_SLOT_BYTES <= CAUSAL_MAILBOX_BYTES,
              "causal mailbox layout exceeds its fixed workspace reservation");

inline constexpr bool EnableCausalSkip(uint32_t, uint64_t seqK)
{
    return seqK >= CAUSAL_SKIP_MIN_SEQ_K;
}

// 此结构保持 POD 格式，使 TTK Kernel 模式无需 GE/op-host 即可构造数据。
struct alignas(8) RegionScoreTilingData {
    uint32_t seqQ;
    // 权重类型编码为 0=BF16、1=FP16、2=FP32，使用前扩展为 FP32。
    uint32_t weightDtype;
    // R 参与 GM 地址且不限制为 4096；逻辑长度使用 64 位，本地 tile 偏移仍用有界 32 位。
    uint64_t seqK;
    uint32_t groups;
    uint32_t usedCubeCores;

    uint32_t qTile;       // 保留元数据，共享计算流程自行确定 Q tile。
    uint32_t kTile;       // 保留元数据，共享计算流程使用 K macro 512。
    uint32_t headNum;     // head 数范围为 4..128，且为 4 的倍数。
    uint32_t headDim;     // v1 版本固定为 256。

    uint32_t maskMode;    // 支持模式为 0 或 3。
    float headScale;
    uint32_t nTile;       // 保留元数据，共享计算流程使用 N tile 128。
    // TopK 选择配置：位 [15:0] 为 topK，位 16 为 ID 排序，位 17 为 scratch 对齐，复用保留字保持 ABI 兼容。
    uint32_t selectionConfig;
    uint32_t qShardCount;  // 共享二维计划的 Q 范围分片数。
    uint32_t kShardCount;  // 每个 Q 分片的 K 范围分片数。

    // 快速布局的元素步长，由 TTK 适配器或 Host 校验。
    uint64_t qStrideG;
    uint64_t qStrideT;
    uint64_t weightStrideG;
    uint64_t weightStrideT;
    uint64_t outStrideG;
    uint64_t outStrideT;
    uint64_t lengthStrideG; // 相对 Kernel workspace 指针的 Scratch 字节偏移。
    uint64_t lengthStrideT;
    // 算子自有数据在平台 SyncAll 前缀之后的字节偏移。
    uint64_t userWorkspaceOffset;
    uint32_t regionSize;
    uint32_t numRequests;
    // Zero batchTokens preserves the full-Score ABI behavior. Runtime batch
    // coordinates always refer to the original global Q/request positions.
    uint32_t batchTokens;
    uint32_t batchBegin;
    uint32_t batchCount;
    uint32_t batchReserved;
    uint64_t scorePoolOffset; // bytes relative to the kernel workspace pointer
};

static_assert(sizeof(RegionScoreTilingData) % 8 == 0, "tiling blob must be naturally aligned");
static_assert(sizeof(RegionScoreTilingData) == 168, "update every tiling-data packer when the ABI changes");

} // 命名空间 IndexerRegionScore

// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。


namespace IndexerRegionScorePaged {
constexpr uint64_t PAGE_LOAD_ALIGNED = 0U;
constexpr uint64_t PAGE_LOAD_WITH_OFFSET = 1U;
// Host 与设备端共用此 POD，所有步长均以元素计。
struct alignas(8) PagedTilingData {
    uint32_t totalTokens;
    uint32_t groups;
    uint32_t headsPerGroup;
    uint32_t proxyDim;
    uint32_t numRequests;
    uint32_t pageCols;
    uint32_t summariesPerPage;
    uint32_t seqK;                 // 逻辑 R 长度，不超过 pageCols*summariesPerPage，范围为 uint32。
    uint32_t qTile;                // Q 范围分片的 token 粒度。
    uint32_t kTile;
    uint32_t usedCubeCores;
    uint32_t maskMode;
    // headScale 不为 1.0 时，在加权 ReLU 归约前缩放所有 head 权重。
    float headScale;
    uint32_t keyPagesPerMacro;
    uint32_t qShardCount;          // 连续 Q 范围分片数。
    uint32_t kShardCount;          // 每个 Q 分片的连续 K 宏分片数。
    // Scalar 页段规划器要求正 block_table 页 ID 小于 paged_key_summary 的物理页数。
    uint32_t numPhysicalPages;
    // 类型编码为 0=FP32、1=BF16、2=FP16；AIV 路径将权重扩展为 FP32。
    uint32_t weightDtype;
    uint64_t qStrideToken;
    uint64_t qStrideGroup;
    uint64_t qStrideHead;
    uint64_t weightStrideToken;
    uint64_t weightStrideGroup;
    uint64_t cacheStridePage;
    uint64_t cacheStrideRegion;
    uint64_t blockTableStrideRequest;
    uint64_t outputStrideGroup;
    uint64_t outputStrideToken;
    uint64_t reservedStride0; // 相对 Kernel workspace 指针的 Scratch 字节偏移。
    // Host 推导的分页搬运模式，非用户属性；0=页对齐，1=支持页内起点。
    uint64_t pageLoadMode;
    uint32_t regionSize;
    // 位 [15:0] 为 topK，位 16 为 ID 排序，位 17 为 scratch 对齐。
    uint32_t selectionConfig;
    // Zero batchTokens preserves the full-Score ABI behavior. Runtime batch
    // coordinates always refer to the original global Q/request positions.
    uint32_t batchTokens;
    uint32_t batchBegin;
    uint32_t batchCount;
    uint32_t batchReserved;
    uint64_t scorePoolOffset; // bytes relative to the kernel workspace pointer
};
static_assert(offsetof(PagedTilingData, pageLoadMode) == 160, "page-load ABI offset");
static_assert(offsetof(PagedTilingData, regionSize) == 168, "region-size ABI offset");
static_assert(offsetof(PagedTilingData, selectionConfig) == 172, "selection ABI offset");
static_assert(sizeof(PagedTilingData) % 8 == 0, "paged tiling must be naturally aligned");
static_assert(offsetof(PagedTilingData, keyPagesPerMacro) == 52, "key-pages ABI offset");
static_assert(sizeof(PagedTilingData) == 200, "update every paged tiling-data packer");
} // 命名空间 IndexerRegionScorePaged

// CANN Tiling 注册宏要求标准 Host 命名空间；此别名供 Host 与设备入口共用。
namespace optiling {
using PagedTilingData = IndexerRegionScorePaged::PagedTilingData;
} // 命名空间 optiling

#endif
