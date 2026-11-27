/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <cstring>
#include "arch35/tiling_plan.h"
#include "host_checks.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../include/arch35/tiling_data_def.h"

// The paged tiler is a declaration-only implementation header.  Do not include
// a .cpp here: op-host sources are compiled as separate translation units.
#include "arch35/paged_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t HEADS_MIN = 4;
constexpr uint32_t HEADS_MAX = 128;
constexpr uint32_t HEADS_ALIGN = 4;
constexpr uint32_t D = 256;
// Keep each Cube micro tile within 128 M rows and split complete Query rows
// between the two AIVs.  The Q macro contains two micro tiles and therefore
// occupies at most 256 FP8 rows in L1 for every supported H.
constexpr uint32_t CUBE_M_ROWS = 128;
constexpr uint32_t K_TILE = 512;
constexpr int64_t MAX_GROUPS = 4096;
constexpr int64_t MAX_SEQ_Q = 1024 * 1024;

uint32_t SelectQMicroTile(uint32_t heads)
{
    const uint32_t queryAlignment = heads % 16U == 0U ? 1U :
                                    heads % 8U == 0U ? 2U : 4U;
    const uint32_t maxQueries = CUBE_M_ROWS / heads;
    const uint32_t alignedQueries = maxQueries / queryAlignment * queryAlignment;
    return std::max(1U, alignedQueries);
}

uint32_t SelectQMacroTile(uint32_t heads, uint32_t qMicroTile)
{
    // A second micro can address Q L1 by 16-row units only when the first
    // micro occupies an integral number of those units. Otherwise keep a
    // one-micro macro and avoid truncating qRowStart / 16 in LoadData.
    return (qMicroTile * heads) % 16U == 0U ? 2U * qMicroTile : qMicroTile;
}

uint32_t GetContinuousWeightDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return 0U;
    }
    if (dtype == ge::DT_FLOAT16) {
        return 1U;
    }
    return 2U; // FP32
}

constexpr uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

bool ContinuousTensorShapesMatch(const gert::Shape &q, const gert::Shape &k, const gert::Shape &w)
{
    return k.GetDim(1) == 1 && k.GetDim(2) == q.GetDim(3) &&
        w.GetDim(0) == q.GetDim(0) && w.GetDim(1) == q.GetDim(1) && w.GetDim(2) == q.GetDim(2);
}

ge::graphStatus ValidateContinuousShapes(gert::TilingContext *context, const gert::Shape &q, const gert::Shape &k,
    const gert::Shape &w, int64_t maxRegionLen, int64_t regionSize)
{
    const int64_t seqQ = q.GetDim(0);
    const int64_t groups = q.GetDim(1);
    const int64_t heads = q.GetDim(2);
    const int64_t dim = q.GetDim(3);
    const int64_t seqK = k.GetDim(0);
    OP_CHECK_IF(groups <= 0 || (groups != 1 && groups % 4 != 0) || seqQ <= 0 || seqK <= 0 ||
                    groups > MAX_GROUPS || seqQ > MAX_SEQ_Q ||
                    // actual_kv_lengths is int32. Keep the logical K extent in
                    // the range that mask_mode=3 can represent even though
                    seqK > INT32_MAX || maxRegionLen < 0 || maxRegionLen > seqK ||
                    regionSize <= 0 || regionSize > INT32_MAX ||
                    heads < HEADS_MIN || heads > HEADS_MAX ||
                    heads % HEADS_ALIGN != 0 || (dim != 128 && dim != D),
                OP_LOGE(context,
                        "shape contract failed: Q[T,G,H,256], K[R,1,256], W[T,G,H], "
                        "G must be 1 or a positive multiple of 4; "
                        "4 <= H <= 128 and H divisible by 4"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ContinuousTensorShapesMatch(q, k, w),
                OP_LOGE(context, "continuous key/weight shape mismatch"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateContinuousLengths(gert::TilingContext *context, int64_t maskMode,
    const gert::StorageShape *lengths, const gert::StorageShape *cuSeqLens, bool topkOnly = false)
{
    (void)topkOnly;
    OP_CHECK_IF(maskMode != 0 && maskMode != 3,
                OP_LOGE(context, "mask_mode must be 0 or 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskMode == 3 && lengths == nullptr,
                OP_LOGE(context, "actual_kv_lengths is required when mask_mode=3"),
                return ge::GRAPH_FAILED);
    if (lengths != nullptr) {
        const auto &l = lengths->GetOriginShape();
        OP_CHECK_IF(maskMode != 3 || l.GetDimNum() != 1 || l.GetDim(0) <= 0,
                    OP_LOGE(context, "actual_kv_lengths must have shape [N] when mask_mode=3"),
                    return ge::GRAPH_FAILED);
        if (cuSeqLens != nullptr) {
            const auto &cu = cuSeqLens->GetOriginShape();
            OP_CHECK_IF(cu.GetDimNum() != 1 || cu.GetDim(0) != l.GetDim(0) + 1,
                        OP_LOGE(context, "cu_seq_lens must have shape [N+1] matching actual_kv_lengths"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(l.GetDim(0) != 1,
                        OP_LOGE(context, "continuous mask_mode=3 without cu_seq_lens supports one request only"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(cuSeqLens != nullptr,
                    OP_LOGE(context, "cu_seq_lens is only valid with continuous mask_mode=3"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateContinuousSelectedOutput(gert::TilingContext *context, int64_t topK, int64_t groups, int64_t seqQ, bool topkOnly = false)
{
    (void)topkOnly;
    const gert::StorageShape *selected = context->GetOutputShape(1);
    OP_CHECK_IF(selected == nullptr,
                OP_LOGE(context, "selected_region_ids output is required"),
                return ge::GRAPH_FAILED);
    const auto &ids = selected->GetOriginShape();
    OP_CHECK_IF(ids.GetDimNum() != 3 || ids.GetDim(0) != groups ||
                    ids.GetDim(1) != seqQ || ids.GetDim(2) != topK,
                OP_LOGE(context, "selected_region_ids must be [G,T,top_k]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfigureContinuousSelectionWorkspace(gert::TilingContext *context, int64_t topK, uint64_t selectedRows,
    size_t libApiWorkspace, size_t *workspace,
    IndexerRegionScore::RegionScoreTilingData &data, bool &useAlignedSelection)
{
    uint32_t rowsPerStoreUnit = 1U;
    while ((rowsPerStoreUnit * topK) % 8 != 0) {
        ++rowsPerStoreUnit;
    }
    constexpr uint64_t kSelectionScratchCapBytes = 16ULL * 1024ULL * 1024ULL;
    const uint64_t paddedK = (static_cast<uint64_t>(topK) + 7U) / 8U * 8U;
    const uint64_t scratchBytesPerRow = paddedK * sizeof(int32_t);
    useAlignedSelection = rowsPerStoreUnit > 1U &&
        (selectedRows + rowsPerStoreUnit - 1U) / rowsPerStoreUnit <
            static_cast<uint64_t>(data.usedCubeCores) * 2U &&
        selectedRows <= kSelectionScratchCapBytes / scratchBytesPerRow;
    if (useAlignedSelection) {
        OP_CHECK_IF(selectedRows > (SIZE_MAX - 31U) / (paddedK * sizeof(int32_t)),
                    OP_LOGE(context, "selection scratch size overflow"), return ge::GRAPH_FAILED);
        const size_t alignedOffset = (workspace[0] + 31U) & ~static_cast<size_t>(31U);
        OP_CHECK_IF(alignedOffset < workspace[0] ||
                        selectedRows * paddedK * sizeof(int32_t) > SIZE_MAX - alignedOffset,
                    OP_LOGE(context, "selection scratch offset overflow"), return ge::GRAPH_FAILED);
        // The kernel workspace pointer already skips libApiWorkspace.
        data.lengthStrideG = alignedOffset - libApiWorkspace;
        workspace[0] = alignedOffset + selectedRows * paddedK * sizeof(int32_t);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateContinuousSelectionConfig(gert::TilingContext *context, int64_t topK, int64_t sortMode,
    int64_t maxRegionLen, int64_t groups, int64_t seqQ)
{
    OP_CHECK_IF(topK <= 0 || topK > 4096 || sortMode != 0 ||
                    (maxRegionLen > 16777216 ||
                     static_cast<uint64_t>(groups) * static_cast<uint64_t>(seqQ) > UINT32_MAX),
                OP_LOGE(context, "invalid TopK/Sort configuration or row/region extent"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateContinuousScoreOutput(gert::TilingContext *context, const gert::Shape &out, int64_t groups,
    int64_t seqQ, int64_t maxRegionLen)
{
    OP_CHECK_IF(out.GetDimNum() != 3 || out.GetDim(0) != groups || out.GetDim(1) != seqQ ||
                    out.GetDim(2) < maxRegionLen,
                OP_LOGE(context, "output must have shape [G,T,R_out] with R_out >= max_region_len"),
                return ge::GRAPH_FAILED);
    const int64_t outputRegions = out.GetDim(2);
    OP_CHECK_IF(outputRegions > UINT32_MAX,
                OP_LOGE(context, "TopK output score stride exceeds uint32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus FiIndexerRegionScoreTiling(gert::TilingContext *context)
{
    const gert::RuntimeAttrs *entryAttrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, entryAttrs);
    const bool *returnScore = entryAttrs->GetAttrNum() > 7U
        ? entryAttrs->GetAttrPointer<bool>(7) : nullptr;
    const bool topkOnly = returnScore != nullptr && !*returnScore;
    const int64_t *cacheMode = entryAttrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheMode);
    OP_CHECK_IF(*cacheMode != 0 && *cacheMode != 1,
                OP_LOGE(context, "cache_mode must be 0 or 1"), return ge::GRAPH_FAILED);
    if (*cacheMode == 1) {
        return IndexerRegionScorePagedTiling(context);
    }
    OP_CHECK_IF(IndexerRegionScoreHost::ReadGraphKeyLayout(entryAttrs).Enabled(),
                OP_LOGE(context, "graph Key layout attrs require paged mode"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(4) != nullptr,
                OP_LOGE(context, "cache_mode=0 does not accept block_table"), return ge::GRAPH_FAILED);
    const int64_t *maskMode = entryAttrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskMode);
    const gert::StorageShape *query = context->GetInputShape(0);
    const gert::StorageShape *weights = context->GetInputShape(1);
    const gert::StorageShape *keySummary = context->GetInputShape(2);
    // Optional inputs are addressed by OpDef IR index, including absent slots.
    const gert::StorageShape *lengths = context->GetOptionalInputShape(3);
    const gert::StorageShape *cuSeqLens = context->GetOptionalInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    OP_CHECK_NULL_WITH_CONTEXT(context, keySummary);
    OP_CHECK_NULL_WITH_CONTEXT(context, weights);
    const gert::StorageShape *output = context->GetOutputShape(0);

    // Outer normalization guarantees dense compute views; physical storage may
    // still be a flat backing allocation with a nonzero logical view offset.
    // Both output modes therefore use the same logical origin dimensions.
    const auto &q = query->GetOriginShape();
    const auto &k = keySummary->GetOriginShape();
    const auto *strides = context->GetRequiredInputStride(2);
    if (strides == nullptr) {
        strides = context->GetInputStride(2);
    }
    // TensorV2 的空步幅表示隐式连续张量。
    // InputIsView 标识 TensorV2 元数据，不代表物理存储不连续。
    const bool implicitContiguousStride = strides != nullptr && strides->GetDimNum() == 0;
    if (!implicitContiguousStride && (strides != nullptr || context->InputIsView(2))) {
        OP_CHECK_IF(strides == nullptr || strides->GetDimNum() != 3 || k.GetDimNum() != 3 ||
                        (k.GetDim(0) > 1 && strides->GetStride(0) != k.GetDim(2)) ||
                        strides->GetStride(2) != 1,
                    OP_LOGE(context, "cache_mode=0 graph key must be contiguous; materialize the view first"),
                    return ge::GRAPH_FAILED);
    }
    const auto &w = weights->GetOriginShape();
    const auto *weightsDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    const ge::DataType weightsDtype = weightsDesc->GetDataType();
    const auto *queryDesc = context->GetInputDesc(0);
    const auto *keyDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    OP_CHECK_IF(!SupportedQkDtype(queryDesc->GetDataType()) ||
                    queryDesc->GetDataType() != keyDesc->GetDataType(),
                OP_LOGE(context, "query and continuous key dtype must match (FP8, FP16 or BF16)"),
                return ge::GRAPH_FAILED);
    const uint32_t K_TILE = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 512U : 256U;
    OP_CHECK_IF(!SupportedWeightDtype(weightsDtype),
                OP_LOGE(context, "weights must be FP16, BF16 or FP32"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(q.GetDimNum() != 4 || k.GetDimNum() != 3 || w.GetDimNum() != 3,
                OP_LOGE(context, "IndexerRegionScore rank mismatch: Q=%zu K=%zu W=%zu bounded=%d",
                    q.GetDimNum(), k.GetDimNum(), w.GetDimNum(), static_cast<int>(topkOnly)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN &&
                    q.GetDim(3) != 256,
                OP_LOGE(context, "FP8 query supports D=256; 16-bit query supports D=128/256"),
                return ge::GRAPH_FAILED);
    const int64_t seqQ = q.GetDim(0);
    const int64_t groups = q.GetDim(1);
    const int64_t heads = q.GetDim(2);
    const int64_t seqK = k.GetDim(0);
    const float *headScale = entryAttrs->GetAttrPointer<float>(1);
    const int64_t *maxRegionLenAttr = entryAttrs->GetAttrPointer<int64_t>(3);
    const int64_t *regionSize = entryAttrs->GetAttrPointer<int64_t>(4);
    const int64_t *topK = entryAttrs->GetAttrPointer<int64_t>(5);
    const int64_t *sortMode = entryAttrs->GetAttrPointer<int64_t>(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, headScale);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRegionLenAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, regionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, topK);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortMode);
    const int64_t maxRegionLen = *maxRegionLenAttr == -1 ? seqK : *maxRegionLenAttr;
    if (ValidateContinuousSelectionConfig(context, *topK, *sortMode, maxRegionLen, groups, seqQ) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateContinuousShapes(context, q, k, w, maxRegionLen, *regionSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateContinuousLengths(context, *maskMode, lengths, cuSeqLens, topkOnly) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto &out = output->GetOriginShape();
    if (!topkOnly && ValidateContinuousScoreOutput(context, out, groups, seqQ, maxRegionLen) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t outputRegions = topkOnly ? ((maxRegionLen + 7) / 8 * 8) : out.GetDim(2);
    OP_CHECK_IF(topkOnly && (out.GetDimNum() != 1 || out.GetDim(0) != 1),
                OP_LOGE(context, "TopK-only Score placeholder must have shape [1]"), return ge::GRAPH_FAILED);
    if (ValidateContinuousSelectedOutput(context, *topK, groups, seqQ, topkOnly) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(context, "invalid core count"), return ge::GRAPH_FAILED);
    // One mixed task consumes one AIC and two AIVs.  Use the hardware-reported
    // capacity rather than a product-specific constant.
    const uint32_t scheduleCubeCores = std::min(aicNum, aivNum / 2U);
    const uint32_t qMicroTile = SelectQMicroTile(static_cast<uint32_t>(heads));
    const uint32_t qMacroTile = SelectQMacroTile(static_cast<uint32_t>(heads), qMicroTile);
    const uint32_t qMacros = (static_cast<uint32_t>(seqQ) + qMacroTile - 1U) / qMacroTile;
    const uint64_t kMacros64 = (static_cast<uint64_t>(maxRegionLen) + K_TILE - 1U) / K_TILE;
    OP_CHECK_IF(kMacros64 > UINT32_MAX,
                OP_LOGE(context, "ceil(R / K_TILE) exceeds uint32 macro index range"),
                return ge::GRAPH_FAILED);
    const uint32_t kMacros = static_cast<uint32_t>(kMacros64);
    // Compile-time cache and mask strategies are encoded in the key.
    const uint32_t weightDtype = GetContinuousWeightDtype(weightsDtype);
    uint64_t tilingKey = IndexerRegionScore::MakeTilingKey(
        0U, static_cast<uint32_t>(*maskMode));
    const bool causalH4Tail = *maskMode == 3 && heads == 4 &&
        static_cast<uint64_t>(seqQ) >= (groups == 1 ? 32U :
            (16U + static_cast<uint64_t>(groups) - 1U) / static_cast<uint64_t>(groups)) &&
        maxRegionLen >= 129 && (maxRegionLen % 128) != 0;
    if (causalH4Tail) {
        tilingKey |= IndexerRegionScore::TILING_KEY_H4_TAIL;
    }
    context->SetTilingKey(tilingKey);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    const size_t libApiWorkspace = platform.GetLibApiWorkSpaceSize();
    // Causal K ends are streamed through a per-CV-pair ping-pong mailbox.
    // Reserve the same bounded region on 28/32/36-core platforms instead of
    // materializing one entry for every (group,Qmacro).
    const bool enableCausalSkip = *maskMode == 3 &&
        IndexerRegionScore::EnableCausalSkip(static_cast<uint32_t>(seqQ),
                                              static_cast<uint64_t>(maxRegionLen));
    const uint64_t causalMailboxBytes = enableCausalSkip
        ? IndexerRegionScore::CAUSAL_MAILBOX_BYTES
        : 0U;
    OP_CHECK_IF(causalMailboxBytes > static_cast<uint64_t>(SIZE_MAX) - libApiWorkspace,
                OP_LOGE(context, "operator workspace size overflow"), return ge::GRAPH_FAILED);
    workspace[0] = libApiWorkspace + static_cast<size_t>(causalMailboxBytes);

    IndexerRegionScore::RegionScoreTilingData data{};
    data.seqQ = static_cast<uint32_t>(seqQ);
    data.weightDtype = weightDtype;
    data.seqK = static_cast<uint64_t>(maxRegionLen);
    data.groups = static_cast<uint32_t>(groups);
    data.usedCubeCores = 1U;
    data.qTile = qMacroTile;
    data.kTile = K_TILE;
    data.headNum = static_cast<uint32_t>(heads);
    data.headDim = static_cast<uint32_t>(q.GetDim(3));
    data.maskMode = static_cast<uint32_t>(*maskMode);
    data.headScale = *headScale;
    // The shared compute path fixes its physical N tile at 128. Keep the POD
    // field populated for binary layout stability; the device does not read it.
    data.nTile = 128U;
    // Enumerate a true 2-D QxK plan for the shared implementation.
    const uint32_t groupTile = SelectPagedGroupTile(
        static_cast<uint32_t>(groups), static_cast<uint32_t>(heads));
    const uint32_t sharedQMicro =
        CUBE_M_ROWS / (groupTile * static_cast<uint32_t>(heads));
    const uint32_t sharedQRanges =
        CeilDiv(static_cast<uint32_t>(seqQ), sharedQMicro);
    const QkShardPlan sharedShardPlan = SelectQkShardPlan(
        sharedQRanges, kMacros, scheduleCubeCores,
        static_cast<uint32_t>(groups), K_TILE, sharedQMicro,
        static_cast<uint32_t>(heads), false, K_TILE);
    data.qShardCount = sharedShardPlan.qShards;
    data.kShardCount = sharedShardPlan.kShards;
    data.usedCubeCores = data.qShardCount * data.kShardCount;
    // A compact odd-K output otherwise forces eight adjacent rows onto one
    // AIV to avoid overlapping 32-byte GM stores.  Let independent AIVs
    // select into aligned private rows, then compact after a global barrier.
    bool useAlignedSelection = false;
    if (!topkOnly && ConfigureContinuousSelectionWorkspace(context, *topK, static_cast<uint64_t>(groups) * seqQ, libApiWorkspace, workspace, data, useAlignedSelection) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(
        platform.CalcTschBlockDim(data.usedCubeCores * 2U, aicNum, aivNum));
    context->SetScheduleMode(1);
    data.qStrideG = static_cast<uint64_t>(heads) * data.headDim;
    data.qStrideT = static_cast<uint64_t>(groups) * heads * data.headDim;
    data.weightStrideG = static_cast<uint64_t>(heads);
    data.weightStrideT = static_cast<uint64_t>(groups) * heads;
    data.outStrideG = static_cast<uint64_t>(seqQ) * outputRegions;
    data.outStrideT = outputRegions;
    data.regionSize = static_cast<uint32_t>(*regionSize);
    data.selectionConfig = static_cast<uint32_t>(*topK) |
        IndexerRegionScore::kSelectionRegionIdOrder |
        (static_cast<uint32_t>(useAlignedSelection) << 17U);
    data.numRequests = lengths == nullptr ? 1U : static_cast<uint32_t>((lengths->GetOriginShape()).GetDim(0));
    data.lengthStrideT = 0;
    data.userWorkspaceOffset = static_cast<uint64_t>(libApiWorkspace);
    if (topkOnly) {
        IndexerRegionScoreHost::BoundedWorkspacePlan bounded;
        OP_CHECK_IF(!IndexerRegionScoreHost::PlanBoundedScoreWorkspace(
                        IndexerRegionScoreHost::kTopKOnlyWorkspaceBudgetBytes, libApiWorkspace, causalMailboxBytes,
                        data.groups, data.seqQ, static_cast<uint32_t>(maxRegionLen),
                        static_cast<uint32_t>(*topK), sharedQMicro, data.numRequests, data.usedCubeCores, bounded),
                    OP_LOGE(context, "internal workspace budget cannot fit the minimum TopK-only batch"),
                    return ge::GRAPH_FAILED);
        data.batchTokens = bounded.batchTokens;
        data.scorePoolOffset = bounded.scorePoolOffset;
        data.lengthStrideG = bounded.scratchOffset;
        data.outStrideT = bounded.regionStride;
        data.outStrideG = uint64_t(data.batchTokens) * bounded.regionStride;
        data.selectionConfig = static_cast<uint32_t>(*topK) |
            IndexerRegionScore::kSelectionRegionIdOrder |
            (static_cast<uint32_t>(*topK % 8 != 0) << 17U);
        workspace[0] = static_cast<size_t>(bounded.totalBytes);
    }

    OP_LOGI(context->GetNodeName(),
            "native-fp8 tiling: G=%u T=%u R=%lu mask=%u key=%llu aic=%u aiv=%u "
            "Qmacro=%u Kmacro=%u NR=%u qMacros=%u kMacros=%u qShards=%u kShards=%u",
            data.groups, data.seqQ, data.seqK, data.maskMode,
            static_cast<unsigned long long>(tilingKey),
            aicNum, aivNum, data.qTile, data.kTile, data.nTile, qMacros, kMacros,
            data.qShardCount, data.kShardCount);

    if (!topkOnly) {
        const uint64_t planBytes = IndexerRegionScore::ScorePlanWorkspaceBytes(data.numRequests, data.usedCubeCores, data.seqQ, data.batchTokens);
        OP_CHECK_IF(planBytes > SIZE_MAX - workspace[0],
                    OP_LOGE(context, "score plan workspace size overflow"), return ge::GRAPH_FAILED);
        workspace[0] += static_cast<size_t>(planBytes);
    }
    auto *raw = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_CHECK_IF(raw->GetCapacity() < sizeof(data), OP_LOGE(context, "tiling buffer too small"),
                return ge::GRAPH_FAILED);
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
