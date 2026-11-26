"""Native Torch/ACLNN public interfaces retain verified AOT execution."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_soc("Ascend950")


@pytest.fixture(scope="module")
def aclnn_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages

    assert torch.npu.is_available() and torch.npu.device_count() == 8
    root = tmp_path_factory.mktemp("native-aclnn")
    build_adapter(root / "adapter")
    build_production_packages(output_dir=root / "release",
                              cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", root / "cache"))
    return root


@pytest.mark.real_compile
@pytest.mark.parametrize("case,dtype", [("matmul", "bfloat16"), ("attention", "float16"), ("attention", "bfloat16")])
def test_native_aclnn_public_interfaces(aclnn_release, case, dtype):
    run_isolated(aclnn_release, "public", case, dtype)


@pytest.mark.real_compile
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("first", ["matmul", "attention"])
def test_torchnpu_queue_order_and_two_production_packages(aclnn_release, mode, first):
    run_isolated(aclnn_release, "queue", str(mode), first)


def run_isolated(release, *args):
    environment = dict(os.environ, FLASHINFER_AOT_PATHS=str(release / "release"),
                       FLASHINFER_ACLNN_ADAPTER_PATH=str(release / "adapter"), FLASHINFER_NPU_JIT_DISABLE="1")
    environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    environment.pop("ASCEND_LAUNCH_BLOCKING", None)
    subprocess.run([sys.executable, "-B", "-m", __spec__.name, *args], cwd=Path(__file__).resolve().parents[2],
                   env=environment, check=True, timeout=180)


def check_public(case, dtype="float16"):
    import torch
    import torch_npu  # noqa: F401

    import flashinfer
    from flashinfer.aclnn import _native
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.runtime import Backend

    torch.manual_seed(20261027)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        if case == "matmul":
            ac = (torch.randn(17, 48) * .25).bfloat16()
            bc = (torch.randn(49, 32) * .25).bfloat16()
            biasc = (torch.randn(33) * .1).bfloat16()
            expected = torch.nn.functional.silu(ac[1:].double() @ bc[1:].double() + biasc[1:].double())
            a, b, bias = ac.npu()[1:], bc.npu()[1:], biasc.npu()[1:]
            python_result = flashinfer.matmul_bias_silu(
                a, b, bias, backend=Backend.ASCENDC, kernel_id=MATMUL_BIAS_SILU_KERNEL_ID)
            result = torch.ops.flashinfer.matmul_bias_silu(
                a, b, bias, backend="ascendc", kernel_id=MATMUL_BIAS_SILU_KERNEL_ID)
            out = torch.empty_like(result)
            assert torch.ops.flashinfer.matmul_bias_silu_out(a, b, bias, out) is None
            assert flashinfer.matmul_bias_silu(a, b, bias, out=out) is out
            atol = rtol = .008
        elif case == "attention":
            dtype = getattr(torch, dtype)
            qc = (torch.rand(129, 2, 128) * 2 - 1).to(dtype)
            kc = (torch.rand(257, 2, 128) * 2 - 1).to(dtype)
            vc = (torch.rand(257, 2, 128) * 2 - 1).to(dtype)
            query, key, value = (t[1:].double().transpose(0, 1) for t in (qc, kc, vc))
            expected = (torch.softmax(query @ key.transpose(1, 2) / 128**.5, -1) @ value).transpose(0, 1)
            q, k, v = qc.npu()[1:], kc.npu()[1:], vc.npu()[1:]
            python_result = flashinfer.single_prefill_with_kv_cache_catlass(
                q, k, v, backend=Backend.ASCENDC, kernel_id=FLASH_ATTENTION_KERNEL_ID)
            result = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(
                q, k, v, backend="ascendc", kernel_id=FLASH_ATTENTION_KERNEL_ID)
            out = torch.empty_like(result)
            assert torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out(q, k, v, out) is None
            assert flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=out) is out
            atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
        else:
            raise ValueError(case)
    torch.npu.current_stream().wait_stream(stream)
    for tensor in (python_result, result, out):
        torch.testing.assert_close(tensor.cpu().double(), expected, atol=atol, rtol=rtol)
    assert len(_native._owners) == 1
    assert torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer_aclnn::_execute", "PrivateUse1")
    assert result.dtype == out.dtype
    assert flashinfer.norm._torch_library_ops is None


def check_queue(mode, first):
    import torch
    import torch_npu

    import flashinfer
    from flashinfer.aclnn import _native
    from flashinfer.aclnn._ops import ensure_registered
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.runtime import get_global_registry

    torch_npu.npu.set_task_queue_enable(mode)
    assert torch_npu.npu.get_task_queue_enable() == mode
    ensure_registered()
    torch.manual_seed(20261028)
    ac = (torch.randn(16, 48) * .2).bfloat16()
    bc = (torch.randn(48, 32) * .2).bfloat16()
    biasc = (torch.randn(32) * .05).bfloat16()
    qc = (torch.rand(128, 2, 128) * .4 - .2).half()
    kc = (torch.rand(256, 2, 128) * .4 - .2).half()
    vc = (torch.rand(256, 2, 128) * .4 - .2).half()
    expected_mm = torch.nn.functional.silu((ac + .125).double() @ bc.double() + biasc.double())
    query = (qc + .125).double().transpose(0, 1)
    key, value = (t.double().transpose(0, 1) for t in (kc, vc))
    expected_fa = (torch.softmax(query @ key.transpose(1, 2) / 128**.5, -1) @ value).transpose(0, 1)
    stream = torch.npu.Stream()
    observations = []
    with torch.npu.stream(stream):
        a, b, bias = ac.npu(), bc.npu(), biasc.npu()
        q, k, v = qc.npu(), kc.npu(), vc.npu()
        order = ("matmul", "attention") if first == "matmul" else ("attention", "matmul")
        for repeat in range(3):
            for case in order:
                if case == "matmul":
                    # These producers/consumers use the ordinary TorchNPU queue.
                    produced = a + .125
                    output = torch.ops.flashinfer.matmul_bias_silu(produced, b, bias)
                    consumed = output + .25
                    expected = expected_mm.bfloat16() + .25
                    atol = rtol = .008
                else:
                    produced = q + .125
                    output = flashinfer.single_prefill_with_kv_cache_catlass(produced, k, v)
                    consumed = output + .25
                    expected = expected_fa.half() + .25
                    atol = rtol = .001
                observations.append((consumed, expected, atol, rtol))
            if repeat == 0:
                assert len(_native._owners) == 2

                def forbidden(*args, **kwargs):
                    raise AssertionError("warm native calls must reuse prepared bindings")

                _native.prepare = forbidden
    torch.npu.current_stream().wait_stream(stream)
    for observed, expected, atol, rtol in observations:
        actual = observed.cpu()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    registry = get_global_registry()
    assert {registry.get(key).runner.provenance.source for key in
            (FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID)} == {"aot"}


if __name__ == "__main__":
    import torch
    import torch_npu

    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    mode, first, second = sys.argv[1:]
    if mode == "queue":
        check_queue(int(first), second)
    elif mode == "public":
        check_public(first, second)
    else:
        raise ValueError(mode)
