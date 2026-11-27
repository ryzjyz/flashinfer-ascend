// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_KEY_LAYOUT_H
#define INDEXER_REGION_SCORE_KEY_LAYOUT_H
#include <cstddef>
#include <cstdint>
#include <limits>
namespace IndexerRegionScoreHost {
// Element strides. Singleton dimensions do not constrain their unused stride.
inline bool IsPagedInnerContiguous(int64_t pages, int64_t summaries, int64_t dim,
    int64_t pageStride, int64_t regionStride, int64_t dimStride)
{
    if (pages <= 0 || summaries <= 0 || dim <= 0 || dimStride != 1 ||
        (summaries > 1 && regionStride != dim) || summaries > INT64_MAX / dim) {
        return false;
    }
    const int64_t pageElements = summaries * dim;
    return pageStride >= pageElements &&
        (pages == 1 || pageStride <= (INT64_MAX - pageElements) / (pages - 1));
}
// Conservatively restrict zero-copy views to 32-byte aligned page/base
// addresses in this first implementation; materialize other views.
inline bool IsPagedCopyAligned(int64_t pageStride, int64_t offset, int64_t elementBytes)
{
    return elementBytes > 0 && 32 % elementBytes == 0 && offset >= 0 &&
        pageStride % (32 / elementBytes) == 0 && offset % (32 / elementBytes) == 0;
}
inline bool IsPagedSpanValid(int64_t pages, int64_t summaries, int64_t dim,
    int64_t pageStride, int64_t offset, int64_t elementBytes)
{
    if (pages <= 0 || summaries <= 0 || dim <= 0 || pageStride <= 0 || offset < 0 ||
        elementBytes <= 0 || summaries > INT64_MAX / dim) {
        return false;
    }
    const int64_t maxElements = INT64_MAX / elementBytes;
    const int64_t pageElements = summaries * dim;
    return pageElements <= maxElements && offset <= maxElements - pageElements &&
        pages - 1 <= (maxElements - pageElements - offset) / pageStride;
}

// Graph-only descriptor bridge. Physical Key is one contiguous alias covering
// the pitched storage span; these attrs recover its logical paged dimensions.
struct GraphKeyLayout {
    int64_t pages = 0;
    int64_t summaries = 0;
    int64_t pageStride = 0;
    bool Enabled() const { return pages != 0 || summaries != 0 || pageStride != 0; }
};

template <typename Attrs>
inline GraphKeyLayout ReadGraphKeyLayout(const Attrs *attrs)
{
    const auto read = [attrs](size_t index) -> int64_t {
        const auto *value = attrs != nullptr && attrs->GetAttrNum() > index
            ? attrs->template GetAttrPointer<int64_t>(index) : nullptr;
        return value == nullptr ? 0 : *value;
    };
    GraphKeyLayout layout;
    layout.pages = read(8U);
    layout.summaries = read(9U);
    layout.pageStride = read(10U);
    return layout;
}

inline bool ValidGraphKeySpan(const GraphKeyLayout &layout, int64_t dim,
    int64_t spanElements, int64_t elementBytes)
{
    if (layout.pages <= 0 || layout.pages > UINT32_MAX || layout.summaries <= 0 ||
        layout.summaries > 1024 || (dim != 128 && dim != 256) || spanElements <= 0 ||
        !IsPagedInnerContiguous(layout.pages, layout.summaries, dim, layout.pageStride, dim, 1) ||
        !IsPagedCopyAligned(layout.pageStride, 0, elementBytes) ||
        !IsPagedSpanValid(layout.pages, layout.summaries, dim, layout.pageStride, 0, elementBytes)) {
        return false;
    }
    return spanElements == (layout.pages - 1) * layout.pageStride + layout.summaries * dim;
}
} // namespace IndexerRegionScoreHost
#endif
