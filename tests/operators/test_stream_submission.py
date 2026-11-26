"""Mixed backend calls preserve the caller's stream and return asynchronously in queue mode."""

import pytest

import flashinfer
from flashinfer.runtime import current_hardware


def _native_case(torch, x):
    if current_hardware().soc_model == "Ascend950":
        identity = torch.eye(64, dtype=x.dtype).to(x.device)
        bias = torch.zeros_like(x[0])
        return (lambda q: flashinfer.matmul_bias_silu(q, identity, bias),
                lambda q: torch.nn.functional.silu(q.float()).to(q.dtype))

    # Arch22 uses an extended ACLNN binding with a scalar attribute.
    from flashinfer.operators.attention.permute_qkv.cases import golden
    kv = torch.zeros((4, 1, 4, 64), dtype=x.dtype, device=x.device)
    cpu_kv = kv.cpu()
    return (lambda q: flashinfer.permute_qkv(q.view(4, 1, 8, 64), kv, kv, 2).view(-1, 64),
            lambda q: golden(torch, q.view(4, 1, 8, 64), cpu_kv, cpu_kv, 2).view(-1, 64))


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("triton_queue", ["1", "0"], ids=["queued", "direct"])
def test_triton_ascendc_stream_submission(npu_runtime, aot_runtime, mode, triton_queue, monkeypatch):
    torch, torch_npu = npu_runtime
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", triton_queue)
    original_mode = torch_npu.npu.get_task_queue_enable()
    main = torch.npu.current_stream()
    side = torch.npu.Stream()
    cpu_x = ((torch.arange(32 * 64).reshape(32, 64) % 7) - 3).bfloat16()
    x = cpu_x.npu()
    weight = torch.zeros_like(x[0])
    native, golden_native = _native_case(torch, x)

    def normalize(tensor):
        value = tensor.float() + 1
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(tensor.dtype)

    def triton(tensor):
        # residual is temporary and can be released as soon as submission returns.
        flashinfer.norm.gemma_fused_add_rmsnorm(tensor, torch.ones_like(tensor), weight, backend="triton")
        return tensor

    expected = normalize(golden_native(normalize(cpu_x)))
    try:
        torch_npu.npu.set_task_queue_enable(mode)
        assert torch_npu.npu.get_task_queue_enable() == mode
        side.wait_stream(main)
        with torch.npu.stream(side):
            triton(native(triton(x.clone())))
            side.synchronize()  # Exclude lazy compilation, loading and initial allocations.
            if mode and triton_queue == "1":
                large = torch.ones((4096, 4096), dtype=x.dtype, device=x.device)
                buffer = torch.empty_like(large)
                torch.mm(large, large, out=buffer)
                side.synchronize()
                # A pending predecessor after return detects an accidental device wait.
                for depth in (32, 64):
                    for _ in range(depth):
                        torch.mm(large, large, out=buffer)
                    predecessor = side.record_event()
                    pending = not predecessor.query()
                    result = triton(native(triton(x.clone())))
                    if pending and not predecessor.query():
                        break
                else:
                    pytest.fail("the warmed mixed-backend call did not return with a pending predecessor")
            else:
                result = triton(native(triton(x.clone())))
        main.wait_stream(side)
        torch.testing.assert_close(result.cpu().float(), expected.float(), atol=.016, rtol=.016)
    finally:
        torch_npu.npu.set_task_queue_enable(original_mode)
