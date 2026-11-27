/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <cstring>
#include "tiling_plan.h"
#include "key_layout.h"
#include <limits>
#include "../host_checks.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../include/arch35/tiling_data_def.h"

namespace optiling {
namespace {
constexpr uint32_t kWeightsInput = 1U;
constexpr uint32_t kSeqKAttr = 3U;
constexpr uint32_t kRegionSizeAttr = 4U;
constexpr uint32_t kHeadsMin = 4;
constexpr uint32_t kHeadsMax = 128;
constexpr uint32_t kHeadsAlign = 4;
constexpr uint32_t kMaxSummariesPerPage = 1024;
inline bool SupportedWeightDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

inline bool SupportedQkDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

inline uint32_t GetPagedWeightDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return 0U;
    }
    if (dtype == ge::DT_BF16) {
        return 1U;
    }
    return 2U; // FP16
}

using IndexerRegionScoreHost::kKeyTile;
using IndexerRegionScoreHost::kProxyDim;
using IndexerRegionScoreHost::PagedShardPlan;
using IndexerRegionScoreHost::QkShardPlan;
using IndexerRegionScoreHost::SelectPagedGroupTile;
using IndexerRegionScoreHost::SelectPagedRequestShardPlan;
using IndexerRegionScoreHost::SelectQkShardPlan;

ge::graphStatus ValidatePagedShapes(gert::TilingContext *context, const gert::Shape &q, const gert::Shape &w,
    const gert::Shape &cache, const gert::Shape &table, const gert::Shape &cuSeqLensShape)
{
    const int64_t totalTokens = q.GetDim(0);
    const int64_t groups = q.GetDim(1);
    const int64_t heads = q.GetDim(2);
    const int64_t pageCols = table.GetDim(1);
    OP_CHECK_IF(totalTokens <= 0 || groups <= 0 || (groups != 1 && groups % 4 != 0) ||
                    table.GetDim(0) <= 0 || pageCols <= 0 ||
                    heads < kHeadsMin || heads > kHeadsMax || heads % kHeadsAlign != 0 ||
                    (q.GetDim(3) != 128 && q.GetDim(3) != kProxyDim) ||
                    w.GetDim(0) != totalTokens || w.GetDim(1) != groups || w.GetDim(2) != heads ||
                    cache.GetDim(0) <= 0 || cache.GetDim(2) != 1 || cache.GetDim(3) != q.GetDim(3) ||
                    cuSeqLensShape.GetDim(0) != table.GetDim(0) + 1,
                OP_LOGE(context, "paged shape contract: Q[T,G,H,256], W[T,G,H], cache[P,S,1,256], "
                                 "table[N,pageCols], cu_seq_lens[N+1], G=1 or a positive multiple of 4, "
                                 "4 <= H <= 128, H divisible by 4"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePagedLengths(gert::TilingContext *context, int64_t maskMode,
    const gert::StorageShape *actualKvLengths, int64_t requestCount, bool topkOnly = false)
{
    (void)topkOnly;
    OP_CHECK_IF(maskMode == 3 && actualKvLengths == nullptr,
                OP_LOGE(context, "actual_kv_lengths is required when mask_mode is 3"),
                return ge::GRAPH_FAILED);
    if (actualKvLengths != nullptr) {
        const auto &actual = actualKvLengths->GetOriginShape();
        OP_CHECK_IF(maskMode != 3 || actual.GetDimNum() != 1 ||
                        actual.GetDim(0) != requestCount,
                    OP_LOGE(context, "actual_kv_lengths must have shape [N] when mask_mode is 3"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePagedSelectedOutput(gert::TilingContext *context, int64_t topK, int64_t groups, int64_t totalTokens, bool topkOnly = false)
{
    (void)topkOnly;
    const gert::StorageShape *selected = context->GetOutputShape(1);
    OP_CHECK_IF(selected == nullptr,
                OP_LOGE(context, "selected_region_ids output is required"),
                return ge::GRAPH_FAILED);
    const auto &ids = selected->GetOriginShape();
    OP_CHECK_IF(ids.GetDimNum() != 3 || ids.GetDim(0) != groups ||
                    ids.GetDim(1) != totalTokens || ids.GetDim(2) != topK,
                OP_LOGE(context, "selected_region_ids must be [G,T,top_k]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePagedRegionExtent(gert::TilingContext *context, int64_t maskMode, int64_t summariesPerCachePage,
    int64_t maxRegionLen, int64_t regionSize, int64_t pageCols,
    const gert::Shape &out, int64_t groups, int64_t totalTokens, bool topkOnly)
{
    OP_CHECK_IF((maskMode != 0 && maskMode != 3) || summariesPerCachePage <= 0 ||
                    maxRegionLen < 0 || regionSize <= 0 ||
                    pageCols > std::numeric_limits<int64_t>::max() / summariesPerCachePage ||
                    maxRegionLen > pageCols * summariesPerCachePage ||
                    (!topkOnly && (out.GetDimNum() != 3 || out.GetDim(0) != groups ||
                    out.GetDim(1) != totalTokens || out.GetDim(2) < maxRegionLen)) ||
                    (topkOnly && (out.GetDimNum() != 1 || out.GetDim(0) != 1)),
                OP_LOGE(context, "mask_mode must be 0 or 3; key_summary.shape[1] must be positive, "
                                 "0 <= max_region_len <= pageCols*key_summary.shape[1], region_size > 0, "
                                 "and output [G,T,R_out] with R_out >= max_region_len"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfigurePagedSelectionWorkspace(gert::TilingContext *context, int64_t topK,
    uint64_t selectedRows, uint32_t usedCubeCores, size_t libApiWorkspace,
    size_t *workspace, IndexerRegionScorePaged::PagedTilingData &data)
{
    uint32_t rowsPerStoreUnit = 1U;
    while ((rowsPerStoreUnit * topK) % 8 != 0) {
        ++rowsPerStoreUnit;
    }
    constexpr uint64_t kSelectionScratchCapBytes = 16ULL * 1024ULL * 1024ULL;
    const uint64_t paddedK = (static_cast<uint64_t>(topK) + 7U) / 8U * 8U;
    const uint64_t scratchBytesPerRow = paddedK * sizeof(int32_t);
    const bool useAlignedSelection = rowsPerStoreUnit > 1U &&
        (selectedRows + rowsPerStoreUnit - 1U) / rowsPerStoreUnit <
            static_cast<uint64_t>(usedCubeCores) * 2U &&
        selectedRows <= kSelectionScratchCapBytes / scratchBytesPerRow;
    if (useAlignedSelection) {
        OP_CHECK_IF(selectedRows > (SIZE_MAX - 31U) / (paddedK * sizeof(int32_t)),
                    OP_LOGE(context, "selection scratch size overflow"), return ge::GRAPH_FAILED);
        const size_t alignedOffset = (workspace[0] + 31U) & ~static_cast<size_t>(31U);
        OP_CHECK_IF(alignedOffset < workspace[0] ||
                        selectedRows * paddedK * sizeof(int32_t) > SIZE_MAX - alignedOffset,
                    OP_LOGE(context, "selection scratch offset overflow"), return ge::GRAPH_FAILED);
        // The runtime passes the kernel workspace pointer after the library
        // prefix, so encode the scratch offset relative to that pointer.
        data.reservedStride0 = alignedOffset - libApiWorkspace;
        workspace[0] = alignedOffset + selectedRows * paddedK * sizeof(int32_t);
    }
    data.selectionConfig = static_cast<uint32_t>(topK) |
        IndexerRegionScore::kSelectionRegionIdOrder |
        (static_cast<uint32_t>(useAlignedSelection) << 17U);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePagedSelectionConfig(gert::TilingContext *context, int64_t topK, int64_t sortMode,
    int64_t maxRegionLen, int64_t groups, int64_t totalTokens)
{
    OP_CHECK_IF(topK <= 0 || topK > 4096 || sortMode != 0 ||
                    (maxRegionLen > 16777216 ||
                     static_cast<uint64_t>(groups) * static_cast<uint64_t>(totalTokens) > UINT32_MAX),
                OP_LOGE(context, "invalid TopK/Sort configuration or row/region extent"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePagedTilingLimits(gert::TilingContext *context, int64_t pageCols, int64_t summariesPerCachePage,
    int64_t maxRegionLen, int64_t regionSize, int64_t totalTokens, int64_t groups,
    int64_t requestCount)
{
    // cu_seq_lens and actual_kv_lengths are INT32, so both token and logical-K
    // extents must remain representable even though the tiling POD uses uint32.
    OP_CHECK_IF(pageCols > std::numeric_limits<uint32_t>::max() / summariesPerCachePage ||
                    maxRegionLen > std::numeric_limits<int32_t>::max() ||
                    regionSize > std::numeric_limits<int32_t>::max() ||
                    totalTokens > std::numeric_limits<int32_t>::max() ||
                    groups > std::numeric_limits<uint32_t>::max() ||
                    requestCount > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "paged shape exceeds prototype uint32 tiling limits"), return ge::GRAPH_FAILED);
    // Partial first/last pages are clipped by the paged L1 adapter.
    OP_CHECK_IF(summariesPerCachePage > kMaxSummariesPerPage,
                OP_LOGE(context, "key_summary.shape[1] must be in [1, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRegionScorePagedTiling(gert::TilingContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *returnScore = attrs->GetAttrNum() > 7U
        ? attrs->GetAttrPointer<bool>(7) : nullptr;
    const bool topkOnly = returnScore != nullptr && !*returnScore;
    const int64_t *maskMode = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskMode);

    // Fixed OpDef IR indices: Q0, W1, Key2, actual3, table4, cu5.
    const gert::StorageShape *indexQ = context->GetInputShape(0);
    const gert::StorageShape *weights = context->GetInputShape(kWeightsInput);
    const gert::StorageShape *keySummary = context->GetInputShape(2);
    const gert::StorageShape *blockTable = context->GetOptionalInputShape(4);
    const gert::StorageShape *cuSeqLens = context->GetOptionalInputShape(5);
    const gert::StorageShape *actualKvLengths = context->GetOptionalInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, weights);
    OP_CHECK_NULL_WITH_CONTEXT(context, keySummary);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTable);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqLens);
    const auto *weightsDesc = context->GetInputDesc(kWeightsInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    const ge::DataType weightsDtype = weightsDesc->GetDataType();
    const auto *queryDesc = context->GetInputDesc(0);
    const auto *cacheDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheDesc);
    OP_CHECK_IF(!SupportedQkDtype(queryDesc->GetDataType()) ||
                    queryDesc->GetDataType() != cacheDesc->GetDataType(),
                OP_LOGE(context, "query and paged key dtype must match (FP8, FP16 or BF16)"),
                return ge::GRAPH_FAILED);
    const uint32_t kKeyTile = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 512U : 256U;
    OP_CHECK_IF(!SupportedWeightDtype(weightsDtype),
                OP_LOGE(context, "weights must be FP16, BF16 or FP32"), return ge::GRAPH_FAILED);
    const auto &q = indexQ->GetOriginShape();
    const auto &w = weights->GetOriginShape();
    const auto &physicalCache = keySummary->GetOriginShape();
    const auto graphKey = IndexerRegionScoreHost::ReadGraphKeyLayout(attrs);
    gert::Shape bridgedCache;
    if (graphKey.Enabled()) {
        const int64_t elementBytes = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 1 : 2;
        OP_CHECK_IF(q.GetDimNum() != 4 || physicalCache.GetDimNum() != 1 ||
                        !IndexerRegionScoreHost::ValidGraphKeySpan(graphKey, q.GetDim(3),
                            physicalCache.GetDim(0), elementBytes),
                    OP_LOGE(context, "invalid graph Key physical-span bridge"),
                    return ge::GRAPH_FAILED);
        bridgedCache.SetDimNum(4);
        bridgedCache.SetDim(0, graphKey.pages);
        bridgedCache.SetDim(1, graphKey.summaries);
        bridgedCache.SetDim(2, 1);
        bridgedCache.SetDim(3, q.GetDim(3));
    }
    const auto &cache = graphKey.Enabled() ? bridgedCache : physicalCache;
    const auto &table = blockTable->GetOriginShape();
    const auto &cuSeqLensShape = cuSeqLens->GetOriginShape();
    OP_CHECK_IF(q.GetDimNum() != 4 || w.GetDimNum() != 3 || cache.GetDimNum() != 4 ||
                    table.GetDimNum() != 2 || cuSeqLensShape.GetDimNum() != 1,
                OP_LOGE(context, "paged rank contract failed: Q=%zu W=%zu Key=%zu table=%zu cu=%zu bounded=%d",
                    q.GetDimNum(), w.GetDimNum(), cache.GetDimNum(), table.GetDimNum(),
                    cuSeqLensShape.GetDimNum(), static_cast<int>(topkOnly)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN &&
                    q.GetDim(3) != 256,
                OP_LOGE(context, "FP8 query supports D=256; 16-bit query supports D=128/256"),
                return ge::GRAPH_FAILED);
    const int64_t totalTokens = q.GetDim(0);
    const int64_t groups = q.GetDim(1);
    const int64_t heads = q.GetDim(2);
    const int64_t pageCols = table.GetDim(1);
    const int64_t summariesPerCachePage = cache.GetDim(1);
    if (ValidatePagedShapes(context, q, w, cache, table, cuSeqLensShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const float *headScale = attrs->GetAttrPointer<float>(1);
    const int64_t *maxRegionLen = attrs->GetAttrPointer<int64_t>(kSeqKAttr);
    const int64_t *regionSize = attrs->GetAttrPointer<int64_t>(kRegionSizeAttr);
    const int64_t *topK = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *sortMode = attrs->GetAttrPointer<int64_t>(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, headScale);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRegionLen);
    OP_CHECK_NULL_WITH_CONTEXT(context, regionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, topK);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortMode);
    if (ValidatePagedSelectionConfig(context, *topK, *sortMode, *maxRegionLen, groups, totalTokens) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePagedLengths(context, *maskMode, actualKvLengths, table.GetDim(0), topkOnly) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    const auto &out = output->GetOriginShape();
    OP_CHECK_IF(out.GetDimNum() == 3 && out.GetDim(2) > UINT32_MAX,
                OP_LOGE(context, "TopK output score stride exceeds uint32"),
                return ge::GRAPH_FAILED);
    if (ValidatePagedSelectedOutput(context, *topK, groups, totalTokens, topkOnly) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePagedRegionExtent(context, *maskMode, summariesPerCachePage, *maxRegionLen, *regionSize, pageCols, out, groups, totalTokens, topkOnly) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePagedTilingLimits(context, pageCols, summariesPerCachePage, *maxRegionLen, *regionSize, totalTokens, groups, table.GetDim(0)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0 || aivNum < 2U, OP_LOGE(context, "invalid Ascend platform core counts"),
                return ge::GRAPH_FAILED);

    const uint32_t requestCount = static_cast<uint32_t>(table.GetDim(0));
    const uint32_t keyRangeCount =
        (static_cast<uint32_t>(*maxRegionLen) + kKeyTile - 1U) / kKeyTile;
    const uint32_t pairCapacity = aicNum < aivNum / 2U ? aicNum : aivNum / 2U;
    const bool balancedRequestGrid = *maskMode == 3 && q.GetDim(3) == 128 &&
        (queryDesc->GetDataType() == ge::DT_BF16 || queryDesc->GetDataType() == ge::DT_FLOAT16);
    const PagedShardPlan requestPlan = SelectPagedRequestShardPlan(
        static_cast<uint32_t>(totalTokens), requestCount, keyRangeCount, pairCapacity,
        static_cast<uint32_t>(groups), static_cast<uint32_t>(summariesPerCachePage),
        static_cast<uint32_t>(heads), kKeyTile, balancedRequestGrid);
    const uint32_t qMicroTile = requestPlan.qMicroTile;
    const uint32_t queryRangeCount =
        (static_cast<uint32_t>(totalTokens) + qMicroTile - 1U) / qMicroTile;
    QkShardPlan shardPlan = requestPlan.shards;
    const bool native16 = queryDesc->GetDataType() == ge::DT_BF16 ||
        queryDesc->GetDataType() == ge::DT_FLOAT16;
    if (native16) {
        shardPlan = IndexerRegionScoreHost::FillPagedQueryShards(
            shardPlan, queryRangeCount, keyRangeCount, pairCapacity);
    }
    const uint32_t usedCubeCores = shardPlan.qShards * shardPlan.kShards;
    // One mixed task consists of one AIC and two AIVs.  Partition the packed
    // token axis in dynamic micro units satisfying
    // qMicroTile*groupTile*heads <= 128 so large single-request batches can
    // use every physical AIC reported by the platform.  Each device pair
    // intersects its global Q range with cu_seq_lens at runtime;
    // therefore no compute tile crosses a request boundary.
    context->SetBlockDim(platform.CalcTschBlockDim(usedCubeCores * 2U, aicNum, aivNum));
    context->SetScheduleMode(1);
    // Compile-time cache, mask and TopK selection strategies are encoded in the key.
    const uint32_t weightDtype = GetPagedWeightDtype(weightsDtype);
    uint64_t tilingKey = IndexerRegionScore::MakeTilingKey(1U, static_cast<uint32_t>(*maskMode));
    const bool causalH4Tail = *maskMode == 3 && native16 && heads == 4 &&
        static_cast<uint64_t>(totalTokens) >= (groups == 1 ? 32U :
            (16U + static_cast<uint64_t>(groups) - 1U) / static_cast<uint64_t>(groups)) &&
        *maxRegionLen >= 129 &&
        (*maxRegionLen % 128) != 0;
    if (causalH4Tail) {
        tilingKey |= IndexerRegionScore::TILING_KEY_H4_TAIL;
    }
    context->SetTilingKey(tilingKey);
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = platform.GetLibApiWorkSpaceSize();

    IndexerRegionScorePaged::PagedTilingData data{};
    data.totalTokens = static_cast<uint32_t>(totalTokens);
    data.weightDtype = weightDtype;
    data.groups = static_cast<uint32_t>(groups);
    data.headsPerGroup = static_cast<uint32_t>(heads);
    data.proxyDim = static_cast<uint32_t>(q.GetDim(3));
    data.numRequests = requestCount;
    data.pageCols = static_cast<uint32_t>(pageCols);
    data.summariesPerPage = static_cast<uint32_t>(summariesPerCachePage);
    data.seqK = static_cast<uint32_t>(*maxRegionLen);
    data.qTile = qMicroTile;
    data.kTile = kKeyTile;
    data.pageLoadMode = kKeyTile % data.summariesPerPage == 0U ?
        IndexerRegionScorePaged::PAGE_LOAD_ALIGNED : IndexerRegionScorePaged::PAGE_LOAD_WITH_OFFSET;
    data.usedCubeCores = usedCubeCores;
    data.maskMode = static_cast<uint32_t>(*maskMode);
    data.headScale = *headScale;
    data.keyPagesPerMacro = kKeyTile % data.summariesPerPage == 0U ?
        kKeyTile / data.summariesPerPage :
        (kKeyTile + data.summariesPerPage - 1U) / data.summariesPerPage + 1U;
    data.qShardCount = shardPlan.qShards;
    data.kShardCount = shardPlan.kShards;
    data.numPhysicalPages = static_cast<uint32_t>(cache.GetDim(0));
    data.qStrideHead = data.proxyDim;
    data.qStrideToken = static_cast<uint64_t>(groups) * heads * data.proxyDim;
    data.qStrideGroup = static_cast<uint64_t>(heads) * data.proxyDim;
    data.weightStrideToken = static_cast<uint64_t>(groups) * heads;
    data.weightStrideGroup = static_cast<uint64_t>(heads);
    data.cacheStrideRegion = data.proxyDim;
    data.cacheStridePage = static_cast<uint64_t>(data.summariesPerPage) * data.proxyDim;
    const auto *strides = context->GetRequiredInputStride(2);
    if (strides == nullptr) {
        strides = context->GetInputStride(2);
    }
    if (graphKey.Enabled()) {
        // The graph input handler has already rebased the contiguous span to
        // the first logical page. No offset is added to the kernel pointer.
        OP_CHECK_IF(strides != nullptr && strides->GetDimNum() != 0 &&
                        (strides->GetDimNum() != 1 || strides->GetStride(0) != 1),
                    OP_LOGE(context, "graph Key physical span must be contiguous"), return ge::GRAPH_FAILED);
        int64_t offset = context->GetRequiredInputOffset(2);
        if (offset < 0) { offset = context->GetInputOffset(2); }
        const int64_t elementBytes = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 1 : 2;
        OP_CHECK_IF(offset >= 0 && !IndexerRegionScoreHost::IsPagedCopyAligned(graphKey.pageStride,
                        offset, elementBytes),
                    OP_LOGE(context, "graph Key physical-span offset must be 32-byte aligned"),
                    return ge::GRAPH_FAILED);
        data.cacheStridePage = static_cast<uint64_t>(graphKey.pageStride);
    } else if (strides != nullptr || context->InputIsView(2)) {
        OP_CHECK_IF(strides == nullptr || strides->GetDimNum() != 4 ||
                        !IndexerRegionScoreHost::IsPagedInnerContiguous(cache.GetDim(0), cache.GetDim(1),
                            cache.GetDim(3), strides->GetStride(0), strides->GetStride(1), strides->GetStride(3)),
                    OP_LOGE(context, "paged key view requires contiguous page contents and page stride >= S*D"),
                    return ge::GRAPH_FAILED);
        int64_t offset = context->GetRequiredInputOffset(2);
        if (offset < 0) {
            offset = context->GetInputOffset(2);
        }
        // Ordinary non-view descriptors may expose strides but no offset.
        if (offset < 0 && !context->InputIsView(2)) {
            offset = 0;
        }
        const int64_t elementBytes = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 1 : 2;
        OP_CHECK_IF(!IndexerRegionScoreHost::IsPagedCopyAligned(strides->GetStride(0), offset, elementBytes) ||
                        !IndexerRegionScoreHost::IsPagedSpanValid(cache.GetDim(0), cache.GetDim(1), cache.GetDim(3),
                            strides->GetStride(0), offset, elementBytes),
                    OP_LOGE(context, "paged key page stride and view offset must be 32-byte aligned"),
                    return ge::GRAPH_FAILED);
        data.cacheStridePage = static_cast<uint64_t>(strides->GetStride(0));
    }
    data.blockTableStrideRequest = data.pageCols;
    data.outputStrideToken = topkOnly ? (uint64_t(*maxRegionLen) + 7U) / 8U * 8U :
        static_cast<uint64_t>(out.GetDim(2));
    data.outputStrideGroup = static_cast<uint64_t>(data.totalTokens) * data.outputStrideToken;
    data.regionSize = static_cast<uint32_t>(*regionSize);
    if (!topkOnly && ConfigurePagedSelectionWorkspace(context, *topK, static_cast<uint64_t>(groups) * totalTokens, usedCubeCores, platform.GetLibApiWorkSpaceSize(), workspace, data) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (topkOnly) {
        IndexerRegionScoreHost::BoundedWorkspacePlan bounded;
        OP_CHECK_IF(!IndexerRegionScoreHost::PlanBoundedScoreWorkspace(
                        IndexerRegionScoreHost::kTopKOnlyWorkspaceBudgetBytes, platform.GetLibApiWorkSpaceSize(), 0U,
                        data.groups, data.totalTokens, data.seqK,
                        static_cast<uint32_t>(*topK), qMicroTile, data.numRequests, usedCubeCores, bounded),
                    OP_LOGE(context, "internal workspace budget cannot fit the minimum TopK-only batch"),
                    return ge::GRAPH_FAILED);
        data.batchTokens = bounded.batchTokens;
        data.scorePoolOffset = bounded.scorePoolOffset;
        data.reservedStride0 = bounded.scratchOffset;
        data.outputStrideToken = bounded.regionStride;
        data.outputStrideGroup = uint64_t(data.batchTokens) * bounded.regionStride;
        data.selectionConfig = static_cast<uint32_t>(*topK) |
            IndexerRegionScore::kSelectionRegionIdOrder |
            (static_cast<uint32_t>(*topK % 8 != 0) << 17U);
        workspace[0] = static_cast<size_t>(bounded.totalBytes);
    }
    if (!topkOnly) {
        const uint64_t planBytes = IndexerRegionScore::ScorePlanWorkspaceBytes(data.numRequests, usedCubeCores, data.totalTokens, data.batchTokens);
        OP_CHECK_IF(planBytes > SIZE_MAX - workspace[0],
                    OP_LOGE(context, "score plan workspace size overflow"), return ge::GRAPH_FAILED);
        workspace[0] += static_cast<size_t>(planBytes);
    }
    auto *raw = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_CHECK_IF(raw->GetCapacity() < sizeof(data), OP_LOGE(context, "paged tiling buffer too small"),
                return ge::GRAPH_FAILED);
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    OP_LOGI(context->GetNodeName(),
            "paged tiling: T=%u G=%u R=%u Pcols=%u S=%u key=%llu aic=%u aiv=%u usedAic=%u qRanges=%u "
            "kRanges=%u qShards=%u kShards=%u",
            data.totalTokens, data.groups, data.seqK, data.pageCols, data.summariesPerPage,
            static_cast<unsigned long long>(tilingKey),
            aicNum, aivNum, data.usedCubeCores, queryRangeCount, keyRangeCount,
            data.qShardCount, data.kShardCount);
    return ge::GRAPH_SUCCESS;
}
} // namespace
} // namespace optiling
