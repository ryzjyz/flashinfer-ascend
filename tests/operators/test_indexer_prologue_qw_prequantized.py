"""Targeted NZ-direction and independent-helper oracle regressions."""

import pytest

import flashinfer
from flashinfer.operators.attention.indexer_prologue_qw_prequantized import cases

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


# Keep the expected native failure last in this module: on this CANN release its
# asynchronous error path corrupts a subsequent NZ-to-CPU helper read.
def test_wrong_nz_physical_direction_is_rejected(npu_runtime, capfd, monkeypatch):
    """A transposed NZ backing with the expected logical shape must not run."""
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, tokens=3)
    source = torch.randn(4096, 1280).to(torch.bfloat16).npu()
    weight_q, weight_scale = torch_npu.npu_dynamic_quant(source, dst_type=torch.int8)
    wrong = torch.as_strided(torch_npu.npu_format_cast(weight_q, 29),
                             (1280, 4096), (4096, 1), 0)
    assert torch_npu.get_npu_format(wrong) == 29 and wrong.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        flashinfer.indexer_prologue_qw_prequantized(
            args[0], args[1], wrong, weight_scale, args[4], args[5], args[6], **options)
    assert "INT8 FRACTAL_NZ storage shape does not match" in capfd.readouterr().err
