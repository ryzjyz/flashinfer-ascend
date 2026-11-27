/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"
#include "host_checks.h"
#include <algorithm>
#include <limits>
#include "arch35/key_layout.h"

namespace ops {
namespace {
static ge::graphStatus ResolvePagedOutputExtent(gert::InferShapeContext *context,
    const gert::RuntimeAttrs *attrs, const gert::Shape *query, int64_t &maxRegions)
{
    const gert::Shape *pagedKeySummary = context->GetInputShape(2);
    const gert::Shape *blockTable = context->GetOptionalInputShape(4);
    const gert::Shape *cuSeqLens = context->GetOptionalInputShape(5);
    const int64_t *maxRegionLen = attrs->GetAttrPointer<int64_t>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqLens);
    OP_CHECK_NULL_WITH_CONTEXT(context, pagedKeySummary);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTable);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRegionLen);
    const auto graphKey = IndexerRegionScoreHost::ReadGraphKeyLayout(attrs);
    if (graphKey.Enabled()) {
        const auto *queryDesc = context->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
        const int64_t elementBytes = queryDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN ? 1 : 2;
        OP_CHECK_IF(query->GetDimNum() != 4 ||
                        pagedKeySummary->GetDimNum() != 1 ||
                        !IndexerRegionScoreHost::ValidGraphKeySpan(graphKey, query->GetDim(3),
                            pagedKeySummary->GetDim(0), elementBytes),
                    OP_LOGE(context, "invalid graph Key physical-span bridge"),
                    return ge::GRAPH_FAILED);
    }
    const int64_t summariesPerPage = graphKey.Enabled() ? graphKey.summaries : pagedKeySummary->GetDim(1);
    OP_CHECK_IF(query->GetDimNum() != 4 || (!graphKey.Enabled() && pagedKeySummary->GetDimNum() != 4) ||
                    blockTable->GetDimNum() != 2 || summariesPerPage <= 0 || *maxRegionLen < 0,
                OP_LOGE(context, "paged layout expects Q[T,G,H,D], key_summary[P,S,1,D], "
                                 "block_table[N,pageCols], S/max_region_len >= 0"),
                return ge::GRAPH_FAILED);
    const int64_t pageCols = blockTable->GetDim(1);
    OP_CHECK_IF(query->GetDim(0) > std::numeric_limits<int32_t>::max() ||
                    pageCols <= 0 || pageCols > INT64_MAX / summariesPerPage ||
                    *maxRegionLen > std::numeric_limits<int32_t>::max() ||
                    *maxRegionLen > pageCols * summariesPerPage,
                OP_LOGE(context, "invalid paged logical region extent"), return ge::GRAPH_FAILED);
    maxRegions = *maxRegionLen;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeIndexerRegionScore(gert::InferShapeContext *context)
{
    const gert::Shape *query = context->GetInputShape(0);
    gert::Shape *output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *cacheMode = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheMode);
    OP_CHECK_IF(*cacheMode != 0 && *cacheMode != 1,
                OP_LOGE(context, "cache_mode must be 0 or 1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*cacheMode == 0 && context->GetOptionalInputShape(4) != nullptr,
                OP_LOGE(context, "cache_mode=0 does not accept block_table"), return ge::GRAPH_FAILED);
    const bool isPaged = *cacheMode == 1;
    const int64_t *regionSize = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *topK = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *sortMode = attrs->GetAttrPointer<int64_t>(6);
    const bool *returnScore = attrs->GetAttrNum() > 7U
        ? attrs->GetAttrPointer<bool>(7) : nullptr;
    const bool topkOnly = returnScore != nullptr && !*returnScore;
    OP_CHECK_IF(IndexerRegionScoreHost::ReadGraphKeyLayout(attrs).Enabled() && !isPaged,
                OP_LOGE(context, "graph Key layout attrs require paged mode"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, regionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, topK);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortMode);
    OP_CHECK_IF(*regionSize <= 0,
                OP_LOGE(context, "region_size must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*topK <= 0 || *topK > 4096 || *sortMode != 0,
                OP_LOGE(context, "top_k must be in [1,4096] and sort_mode must be 0 (ascending region ID)"),
                return ge::GRAPH_FAILED);
    if (isPaged) {
        int64_t maxRegions = 0;
        if (ResolvePagedOutputExtent(context, attrs, query, maxRegions) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const int64_t outputRegions = output->GetDimNum() == 3 &&
                output->GetDim(0) == query->GetDim(1) && output->GetDim(1) == query->GetDim(0) &&
                output->GetDim(2) >= maxRegions
            ? output->GetDim(2) : std::max<int64_t>(maxRegions, 1);
        output->SetDimNum(3);
        output->SetDim(0, query->GetDim(1));
        output->SetDim(1, query->GetDim(0));
        output->SetDim(2, outputRegions);
        if (topkOnly) { output->SetDimNum(1); output->SetDim(0, 1); }
        gert::Shape *selectedRegionIds = context->GetOutputShape(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, selectedRegionIds);
        selectedRegionIds->SetDimNum(3);
        selectedRegionIds->SetDim(0, query->GetDim(1));
        selectedRegionIds->SetDim(1, query->GetDim(0));
        selectedRegionIds->SetDim(2, *topK);
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape *keySummary = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, keySummary);
    OP_CHECK_IF(query->GetDimNum() != 4 || keySummary->GetDimNum() != 3,
                OP_LOGE(context, "contiguous layout expects query [T,G,H,D] and key_summary [R,1,D]"),
                return ge::GRAPH_FAILED);
    const int64_t *maxRegionLen = attrs->GetAttrPointer<int64_t>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRegionLen);
    const int64_t keyRegions = keySummary->GetDim(0);
    const int64_t effectiveMaxRegionLen = *maxRegionLen == -1 ? keyRegions : *maxRegionLen;
    OP_CHECK_IF(*maxRegionLen < -1 || effectiveMaxRegionLen > keyRegions,
                OP_LOGE(context, "max_region_len must be -1 or in [0, key_summary.shape[0]]"),
                return ge::GRAPH_FAILED);
    const int64_t outputRegions = output->GetDimNum() == 3 &&
            output->GetDim(0) == query->GetDim(1) && output->GetDim(1) == query->GetDim(0) &&
            output->GetDim(2) >= effectiveMaxRegionLen
        ? output->GetDim(2) : std::max<int64_t>(effectiveMaxRegionLen, 1);
    output->SetDimNum(3);
    output->SetDim(0, query->GetDim(1));
    output->SetDim(1, query->GetDim(0));
    output->SetDim(2, outputRegions);
    if (topkOnly) { output->SetDimNum(1); output->SetDim(0, 1); }
    gert::Shape *selectedRegionIds = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedRegionIds);
    selectedRegionIds->SetDimNum(3);
    selectedRegionIds->SetDim(0, query->GetDim(1));
    selectedRegionIds->SetDim(1, query->GetDim(0));
    selectedRegionIds->SetDim(2, *topK);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerRegionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

} // namespace
} // namespace ops

namespace optiling {
ge::graphStatus FiIndexerRegionScoreTiling(gert::TilingContext *context);
}

namespace ops {
class FiIndexerRegionScore : public OpDef {
public:
    explicit FiIndexerRegionScore(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // One required Key: cache_mode=0 dense, cache_mode=1 block-table mapped.
        this->Input("key_summary")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("actual_kv_lengths")
            .ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cu_seq_lens")
            .ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("selected_region_ids")
            .ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("head_scale").AttrType(OPTIONAL).Float(1.0F);
        // Keep the public enum for future layouts; only 0/1 are supported now.
        this->Attr("cache_mode").AttrType(OPTIONAL).Int(0);
        // The number of logical region scores to produce.  In continuous
        // mode -1 means the complete key_summary extent; zero is a valid
        // empty candidate range. Paged mode accepts [0, C*S].
        this->Attr("max_region_len").AttrType(OPTIONAL).Int(-1);
        // Number of uncompressed KV tokens represented by one region.
        this->Attr("region_size").AttrType(OPTIONAL).Int(8);
        // Required selection count [1,4096]; retain the unified attribute order.
        // sort_mode: 0, ascending logical region ID.
        this->Attr("top_k").AttrType(OPTIONAL).Int(512);
        this->Attr("sort_mode").AttrType(OPTIONAL).Int(0);
        // The IDs-only tiler gives the required Score slot a one-element placeholder.
        this->Attr("return_score").AttrType(OPTIONAL).Bool(true);
        // Preserve the page pitch when the runner binds a flat physical Key span.
        this->Attr("graph_key_pages").AttrType(OPTIONAL).Int(0);
        this->Attr("graph_key_summaries").AttrType(OPTIONAL).Int(0);
        this->Attr("graph_key_page_stride").AttrType(OPTIONAL).Int(0);

        SetInferShape(InferShapeIndexerRegionScore).SetInferDataType(InferDataTypeIndexerRegionScore);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opInterface.value", "fi_indexer_region_score")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().SetTiling(optiling::FiIndexerRegionScoreTiling).AddConfig("ascend950", config);
    }
};
OP_ADD(FiIndexerRegionScore);
} // namespace ops
