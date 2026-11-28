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


def _run_cold_process(code, environment, *, timeout=300):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", **environment),
        capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_wrong_nz_direction_is_rejected_in_a_cold_process(tmp_path):
    """Isolate the expected native failure from later CANN operator tests."""
    from flashinfer.operators.pytest_support import prepare_release
    from flashinfer.runtime import current_hardware

    kernel_id = ("ascendc/attention/indexer_prologue_qw_prequantized/"
                 "native/op_package/" + current_hardware().soc_model)
    release = prepare_release(
        "attention.indexer_prologue_qw_prequantized", tmp_path, kernel_id=kernel_id)
    code = r'''
import torch
import torch_npu
import flashinfer
from flashinfer.operators.attention.indexer_prologue_qw_prequantized import cases

torch.npu.set_device(int(__import__('os').environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
torch.npu.config.allow_internal_format = True
args, options, _ = cases.sample(torch, torch_npu, tokens=3)
source = torch.randn(4096, 1280).to(torch.bfloat16).npu()
weight_q, weight_scale = torch_npu.npu_dynamic_quant(source, dst_type=torch.int8)
wrong = torch.as_strided(torch_npu.npu_format_cast(weight_q, 29),
                         (1280, 4096), (4096, 1), 0)
assert torch_npu.get_npu_format(wrong) == 29 and wrong.is_contiguous()
try:
    flashinfer.indexer_prologue_qw_prequantized(
        args[0], args[1], wrong, weight_scale, args[4], args[5], args[6], **options)
except (ValueError, RuntimeError) as error:
    print(str(error))
    print("REJECTED")
else:
    raise SystemExit("wrong NZ direction was accepted")
'''
    result = _run_cold_process(code, dict(
        release, FLASHINFER_NPU_JIT_DISABLE="1", ASCEND_LAUNCH_BLOCKING="1"))
    assert "REJECTED" in result.stdout
    assert "INT8 FRACTAL_NZ storage shape does not match" in result.stdout + result.stderr
