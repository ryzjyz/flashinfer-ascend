"""Independent helper oracle for the caller-prequantized QW contract."""

import pytest

import flashinfer

pytestmark = [pytest.mark.real_compile,
              pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


def test_prepare_weight_matches_independent_cpu_oracle(npu_runtime):
    """The helper is tested against a separate CPU quant/transpose/scale path."""
    import math
    from flashinfer.operators.attention.indexer_prologue_qw import cases as dense_cases

    torch, torch_npu = npu_runtime
    generator = torch.Generator(device="cpu").manual_seed(20261120)
    wqb = dense_cases._unit_amax(
        torch, (torch.randn(4096, 1280, generator=generator) / math.sqrt(1280)).to(torch.bfloat16))
    weight_q, weight_scale = dense_cases._dynamic_quant(torch, wqb)
    nz, scale = flashinfer.prepare_indexer_prologue_qw_weight(wqb.npu())
    logical = nz.detach().cpu().contiguous()
    assert torch.equal(logical.int(), weight_q.t().contiguous().int())
    torch.testing.assert_close(scale.cpu().double(), weight_scale.double(), atol=0, rtol=0)
