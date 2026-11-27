"""Host-only IRS workspace bounds and stable data offsets."""

from pathlib import Path
import shutil
import subprocess


def test_score_plan_is_inside_ids_only_workspace_budget(tmp_path):
    compiler = shutil.which("c++")
    assert compiler, "IRS host workspace validation requires c++"
    root = Path(__file__).resolve().parents[2]
    header = root / "csrc/kernels/sparse_indexer/index_region_score/op_host/arch35/tiling_plan.h"
    source = tmp_path / "workspace.cpp"
    source.write_text(r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include "tiling_plan.h"

using namespace IndexerRegionScoreHost;

void Check(uint64_t budget, uint64_t library, uint64_t aux, uint32_t groups,
    uint32_t tokens, uint32_t regions, uint32_t topK, uint32_t micro,
    uint32_t requests, uint32_t pairs)
{
    BoundedWorkspacePlan plan;
    assert(PlanBoundedScoreWorkspace(budget, library, aux, groups, tokens,
        regions, topK, micro, requests, pairs, plan));
    const uint64_t metadata = IndexerRegionScore::ScorePlanWorkspaceBytes(
        requests, pairs, tokens, plan.batchTokens);
    assert(plan.batchTokens > 0U && plan.batchTokens <= tokens);
    assert(plan.totalBytes <= budget - kBoundedExecutorReserve);
    assert(plan.scratchOffset == ((library + aux + 31U) & ~uint64_t(31U)) - library);
    const uint64_t scratchRow = topK % 8U == 0U ? 0U :
        ((uint64_t(topK) + 7U) & ~uint64_t(7U)) * sizeof(int32_t);
    assert(plan.scorePoolOffset == plan.scratchOffset +
        uint64_t(groups) * plan.batchTokens * scratchRow);
    assert(plan.totalBytes == library + metadata + plan.scorePoolOffset +
        uint64_t(groups) * plan.batchTokens * plan.regionStride * sizeof(float));
    std::printf("T=%u B=%u R=%u K=%u batch=%u epochs=%llu plan=%llu total=%llu budget=%llu\n",
        tokens, requests, regions, topK, plan.batchTokens,
        static_cast<unsigned long long>((uint64_t(tokens) + plan.batchTokens - 1U) / plan.batchTokens),
        static_cast<unsigned long long>(metadata), static_cast<unsigned long long>(plan.totalBytes),
        static_cast<unsigned long long>(budget));
}

int main()
{
    constexpr uint64_t budget = kTopKOnlyWorkspaceBudgetBytes;
    constexpr uint64_t library = 16ULL * 1024ULL * 1024ULL;
    BoundedWorkspacePlan old;
    assert(PlanBoundedWorkspace(budget, library, 0U, 1U, 1000000U, 128U, 512U, 4U, old));
    const uint64_t oldMetadata = IndexerRegionScore::ScorePlanWorkspaceBytes(20U, 28U, 1000000U, old.batchTokens);
    assert(old.totalBytes + oldMetadata > budget);
    Check(budget, library, 0U, 1U, 1000000U, 128U, 512U, 4U, 20U, 28U);
    Check(budget, library, 4096U, 1U, 1000000U, 128U, 513U, 4U, 20U, 28U);
    Check(budget, library, 0U, 1U, std::numeric_limits<uint32_t>::max(), 128U, 512U, 4U, 20U, 28U);
    Check(budget, library, 4096U, 4U, 1048576U, 16383U, 4096U, 8U, 1024U, 36U);
    Check(budget, library, 0U, 1U, 80U, 131072U, 512U, 4U, 20U, 28U);
    Check(budget, library, 0U, 1U, 802U, 100U, 512U, 32U, 1U, 26U);
    Check(budget, library, 0U, 1U, 16384U, 8191U, 513U, 32U, 1U, 28U);
    Check(library + 100U * 512U + kBoundedExecutorReserve, library, 0U,
        1U, 100U, 128U, 512U, 4U, 20U, 28U);
    // A one-token pool exactly at its minimum and a one-byte-short rejection.
    const uint64_t metadata = IndexerRegionScore::ScorePlanWorkspaceBytes(20U, 28U, 1U, 1U);
    const uint64_t minimum = library + metadata + 32U + kBoundedExecutorReserve;
    Check(minimum, library, 0U, 1U, 1U, 0U, 8U, 1U, 20U, 28U);
    BoundedWorkspacePlan failed;
    assert(!PlanBoundedScoreWorkspace(library + metadata + 31U + kBoundedExecutorReserve,
        library, 0U, 1U, 1U, 0U, 8U, 1U, 20U, 28U, failed));
    assert(failed.totalBytes == 0U && failed.batchTokens == 0U);
    assert(!PlanBoundedScoreWorkspace(budget, library, 0U, 1U, 80U, 128U, 512U,
        4U, std::numeric_limits<uint32_t>::max(), 28U, failed));
}
""")
    binary = tmp_path / "workspace"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I" + str(header.parent), str(source), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    print(result.stdout, end="")


def test_index_region_score_aligned_output_checks_storage_not_shape():
    from types import SimpleNamespace

    import torch

    from flashinfer.ascendc.attention.index_region_score.runner import _aligned_score_span

    nd = SimpleNamespace(get_npu_format=lambda tensor: 2)
    groups, tokens, logical, pitch = 4, 5, 257, 384
    backing = torch.empty(groups, tokens, pitch, dtype=torch.float32)
    score = backing[..., :logical]
    padded = _aligned_score_span(score, logical, (), nd)
    assert padded.shape == backing.shape
    assert padded.stride() == backing.stride()
    assert padded.data_ptr() == score.data_ptr()

    # Padding remains part of the native output address range.
    assert _aligned_score_span(score, logical, (backing[..., logical:],), nd) is None
    assert _aligned_score_span(score, logical - 1, (), nd) is None

    # A view with the same logical strides may lack the final row's padding.
    short = torch.empty((groups * tokens - 1) * pitch + logical)
    score = short.as_strided((groups, tokens, logical), (tokens * pitch, pitch, 1))
    assert _aligned_score_span(score, logical, (), nd) is None
