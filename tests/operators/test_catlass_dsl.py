"""Route dense prefill and verify CATLASS DSL launch ordering on Ascend950."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from flashinfer.runtime import (
    Backend, CompileMode, DispatchError, GraphCapability, HardwareSignature,
    InputSignature, dispatch, get_global_registry,
)


OP = "ascendc.single_prefill_catlass"
KERNEL_ID = "catlass_dsl/attention/single_prefill_catlass/Ascend950"


def test_catlass_dsl_explicit_selection_preserves_default_priority(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    monkeypatch.delenv("FLASHINFER_NPU_DISPATCH_ORDER", raising=False)
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")
    signature = InputSignature(
        OP, dtypes=("float16",) * 4, shapes=((128, 2, 128),) * 4,
        physical_layouts=("ND",) * 4, specializations=(("tensor_contract", "supported"),),
    )
    hardware = HardwareSignature("Ascend950", 32)
    selected = dispatch(signature, hardware, backend="catlass_dsl")
    assert selected.kernel_id == KERNEL_ID
    assert selected.backend == Backend.CATLASS_DSL and selected.stack_id == "catlass_dsl"
    assert dispatch(signature, hardware, backend="auto").backend == Backend.ASCENDC
    assert dispatch(signature, hardware, kernel_id=KERNEL_ID).backend == Backend.CATLASS_DSL
    descriptor = get_global_registry().get(selected.kernel_id)
    assert descriptor.compile_mode == CompileMode.JIT
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE

    for soc in ("Ascend910B", "Ascend910C"):
        with pytest.raises(DispatchError, match="catlass_dsl"):
            dispatch(signature, HardwareSignature(soc, 32), backend="catlass_dsl")
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_JIT_DISABLE"):
        dispatch(signature, hardware, backend="catlass_dsl")


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
@pytest.mark.parametrize("queue_mode", [0, 1, 2])
def test_catlass_dsl_stream_storage_and_shape_reuse(queue_mode):
    environment = dict(os.environ, TASK_QUEUE_ENABLE=str(queue_mode), FLASHINFER_NPU_JIT_DISABLE="0")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.operators.test_catlass_dsl", str(queue_mode)],
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CATLASS_DSL PASS queue={queue_mode} compilations=2" in result.stdout


def check_stream(queue_mode):
    import torch
    import torch_npu
    import catlass.tla as tla

    import flashinfer

    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    assert torch_npu.npu.get_task_queue_enable() == queue_mode
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(20261125)
    main = torch.npu.current_stream()
    side = torch.npu.Stream()
    with patch.object(tla, "compile", wraps=tla.compile) as compile_calls:
        for dtype in (torch.float16, torch.bfloat16):
            for s, t, heads in ((128, 256, 1), (512, 384, 8)):
                q, k, v = ((torch.rand(length, heads, 128, generator=rng) * 2 - 1).to(dtype)
                           for length in (s + 1, t, t))
                query = (q + .125)[1:].double().transpose(0, 1)
                key, value = (x.double().transpose(0, 1) for x in (k, v))
                expected = (torch.softmax((query @ key.transpose(1, 2)) / 128**.5, -1)
                            @ value).transpose(0, 1).to(dtype) + .25
                with torch.npu.stream(side):
                    device_q, device_k, device_v = (x.npu() for x in (q, k, v))
                    supplied = torch.empty_like(device_q)[1:]
                    result = flashinfer.single_prefill_with_kv_cache_catlass(
                        (device_q + .125)[1:], device_k, device_v,
                        out=supplied, backend="catlass_dsl",
                    )
                    assert result is supplied
                    consumed = result + .25
                main.wait_stream(side)
                atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
                torch.testing.assert_close(consumed.cpu(), expected, atol=atol, rtol=rtol)
        assert compile_calls.call_count == 2
    with torch.npu.stream(side):
        check_submission((device_q[1:], device_k, device_v), supplied, side, queue_mode)
    print(f"CATLASS_DSL PASS queue={queue_mode} compilations={compile_calls.call_count}")


def check_submission(tensors, out, stream, queue_mode):
    from torch_npu import profiler
    import flashinfer

    with TemporaryDirectory() as trace_dir:
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                              on_trace_ready=profiler.tensorboard_trace_handler(trace_dir)) as profile:
            flashinfer.single_prefill_with_kv_cache_catlass(*tensors, out=out, backend="catlass_dsl")
            stream.synchronize()
            profile.step()
        trace, = Path(trace_dir).rglob("trace_view.json")
        events = json.loads(trace.read_text())

    enqueued = [event for event in events if event.get("name") == "Enqueue@dense_prefill"]
    dequeued = [event for event in events if event.get("name") == "Dequeue@dense_prefill"]
    assert len(enqueued) == len(dequeued) == int(queue_mode != 0)
    if queue_mode:
        assert enqueued[0]["args"]["correlation_id"] == dequeued[0]["args"]["correlation_id"]
        assert enqueued[0]["tid"] != dequeued[0]["tid"]


if __name__ == "__main__":
    check_stream(int(sys.argv[1]))
