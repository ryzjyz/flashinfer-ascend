#ifndef INDEXER_REGION_SCORE_DETAIL_REQUEST_HPP
#define INDEXER_REGION_SCORE_DETAIL_REQUEST_HPP

#include <cstdint>
#include "kernel_operator.h"

namespace IndexerRegionScorePaged {

// actualKvLengths 是包含本批 Query 的有效 KV token 数；减去完整 request 的 Q 数得到零基首位置。
__aicore__ inline int64_t RequestFirstTokenPosition(
    int32_t actualKvLength, uint32_t requestBegin, uint32_t requestEnd)
{
    return static_cast<int64_t>(actualKvLength) -
        static_cast<int64_t>(requestEnd - requestBegin);
}


} // namespace IndexerRegionScorePaged

#endif
