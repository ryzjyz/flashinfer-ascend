#pragma once

#include <cstdio>

#define OP_LOGI(node, ...) ((void)0)
#define OP_LOGE(node, ...) do { std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
#define OP_CHECK_IF(condition, diagnostic, action) \
    do { if (condition) { diagnostic; action; } } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    OP_CHECK_IF((pointer) == nullptr, OP_LOGE(context, "Missing IRS metadata: %s", #pointer), \
        return ge::GRAPH_FAILED)
