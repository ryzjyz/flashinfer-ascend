"""Runner reuse through real public operator entry points."""

from unittest.mock import Mock

import pytest

import flashinfer


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
def test_matmul_out_reuses_selection_with_live_tensors(npu_runtime, aot_runtime, monkeypatch):
    from flashinfer import gemm
    from flashinfer._matmul import AOT_KERNEL_ID, CANN_KERNEL_ID
    from flashinfer.jit.tensor_adapter import TorchOpPackageRunner
    from flashinfer.runtime import DispatchError, clear_dispatch_cache

    torch, torch_npu = npu_runtime
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    clear_dispatch_cache()
    signature = Mock(wraps=gemm.build_signature)
    decisions = []
    dispatch = gemm.dispatch

    def select(*args, **kwargs):
        decision = dispatch(*args, **kwargs)
        decisions.append(decision.kernel_id)
        return decision

    launches = []
    execute = TorchOpPackageRunner.__call__

    def launch(self, a, b, out):
        launches.append(out.data_ptr())
        return execute(self, a, b, out)

    monkeypatch.setattr(gemm, "build_signature", signature)
    monkeypatch.setattr(gemm, "dispatch", select)
    monkeypatch.setattr(TorchOpPackageRunner, "__call__", launch)
    b = torch.eye(16, dtype=torch.bfloat16).npu()
    a = torch.ones_like(b)
    out = torch.empty_like(a)
    flashinfer.matmul(a, b, out=out, backend="ascendc")
    assert signature.call_count == 1 and decisions == [AOT_KERNEL_ID]

    main = torch.npu.current_stream()
    side = torch.npu.Stream()
    side.wait_stream(main)
    with torch.npu.stream(side):
        different = a + 3
        view = torch.empty((17, 16), device=a.device, dtype=a.dtype)[1:]
        torch.ops.flashinfer.matmul_out(different, b, view, backend="ascendc")
        consumed = view + 1
    main.wait_stream(side)
    torch.testing.assert_close(consumed.cpu(), torch.full((16, 16), 5, dtype=a.dtype))
    assert signature.call_count == 1 and decisions == [AOT_KERNEL_ID]
    assert launches == [out.data_ptr(), view.data_ptr()]

    with pytest.raises(ValueError, match="overlap"):
        flashinfer.matmul(a, b, out=a.view_as(out), backend="ascendc")
    assert len(launches) == 2
    for invalid in (a.t(), torch_npu.npu_format_cast(a, 29), a.float()):
        before = signature.call_count
        with pytest.raises(DispatchError):
            flashinfer.matmul(invalid, b, out=out, backend="ascendc")
        assert signature.call_count == before + 1
    assert len(launches) == 2

    bigger = a.repeat(2, 1)
    larger_out = torch.empty_like(bigger)
    flashinfer.matmul(bigger, b, out=larger_out, backend="ascendc")
    torch.testing.assert_close(larger_out, bigger)
    assert decisions == [AOT_KERNEL_ID, AOT_KERNEL_ID]
    with monkeypatch.context() as patch:
        patch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")
        flashinfer.matmul(a, b, out=out, backend="ascendc")
    flashinfer.matmul(a, b, out=out, backend="cann")
    assert decisions[-2:] == [AOT_KERNEL_ID, CANN_KERNEL_ID]
    torch.testing.assert_close(out, a)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
def test_auto_out_follows_registry_and_reselects_after_failed_launch(npu_runtime, aot_runtime, monkeypatch):
    from dataclasses import replace
    from flashinfer import gemm
    from flashinfer._matmul import AOT_KERNEL_ID, CANN_KERNEL_ID
    from flashinfer.runtime import KernelRegistry, get_global_registry, kernel_registry_scope

    torch, _ = npu_runtime
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "auto")
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    original = get_global_registry()
    native = original.get(AOT_KERNEL_ID)
    attempts = 0

    def retry_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first launch failed")
        return native.runner(*args)

    native_only = KernelRegistry([replace(native, runner=retry_once)])
    cann_only = KernelRegistry([original.get(CANN_KERNEL_ID)])
    native_only.freeze()
    cann_only.freeze()
    a = torch.ones((16, 16), dtype=torch.bfloat16, device="npu")
    out = torch.empty_like(a)
    select = Mock(wraps=gemm.dispatch)
    monkeypatch.setattr(gemm, "dispatch", select)
    with kernel_registry_scope(native_only):
        with pytest.raises(RuntimeError, match="first launch failed"):
            flashinfer.matmul(a, a, out=out)
        flashinfer.matmul(a, a, out=out)
        flashinfer.matmul(a, a, out=out)
        assert select.call_count == 2 and attempts == 3
        with kernel_registry_scope(cann_only):
            flashinfer.matmul(a, a, out=out)
        flashinfer.matmul(a, a, out=out)
        assert select.call_count == 3 and attempts == 4
    torch.testing.assert_close(out, torch.full_like(out, 16))


@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
def test_declared_operator_caches_attrs_and_revalidates_values(npu_runtime, monkeypatch):
    from flashinfer.operators import runtime
    from flashinfer.operators.gemm.grouped_matmul_k_dim import cases
    from flashinfer.runtime import clear_dispatch_cache

    torch, torch_npu = npu_runtime
    select = Mock(wraps=runtime.dispatch)
    monkeypatch.setattr(runtime, "dispatch", select)
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    clear_dispatch_cache()
    (x, w, groups), _, gold = cases.sample(torch, torch_npu, m=32, n=32, groups=(16, 16))
    grouped = flashinfer.gemm.grouped_matmul_k_dim
    first = grouped(x, w, groups, backend="reference")
    second = grouped(-x, w, groups, trans_a=False, fp32_output=False, backend="reference")
    torch.testing.assert_close(first.cpu(), gold)
    torch.testing.assert_close(second.cpu(), -gold)
    assert first.data_ptr() != second.data_ptr() and select.call_count == 1

    # Square X keeps all input/output shapes equal while the attribute changes the result.
    transposed = grouped(x, w, groups, trans_a=True, backend="reference")
    expected = cases.golden(torch, x.cpu(), w.cpu(), (16, 32), trans_a=True)
    torch.testing.assert_close(transposed.cpu(), expected)
    assert select.call_count == 2
    buf = torch.empty_like(first)
    for out in (buf, torch.empty_like(buf)):
        assert grouped(x, w, groups, out=out, backend="reference") is out
        torch.testing.assert_close(out.cpu(), gold)
    assert select.call_count == 3

    # The original tensor changes in place after warmup. Its values still get checked.
    groups[-1] = 31
    with pytest.raises(ValueError, match="end at the K extent"):
        grouped(x, w, groups, backend="reference")
    assert select.call_count == 3


@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
def test_domain_operator_caches_optional_input_output_and_dtype(npu_runtime, monkeypatch):
    from flashinfer.runtime import clear_dispatch_cache, runner

    torch, _ = npu_runtime
    select = Mock(wraps=runner.dispatch)
    monkeypatch.setattr(runner, "dispatch", select)
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    clear_dispatch_cache()
    a = torch.ones((16, 16), dtype=torch.bfloat16, device="npu")
    b = torch.eye(16, dtype=a.dtype).to(a.device)
    bias = torch.ones_like(a[0])
    mm = flashinfer.gemm.mm_bf16
    for value in (a, a + 2):
        torch.testing.assert_close(mm(value, b, backend="cann"), value)
    assert select.call_count == 1
    torch.testing.assert_close(mm(a, b, bias=bias, backend="cann"), a + 1)
    assert select.call_count == 2
    torch.testing.assert_close(mm(a, b, bias=bias, out_dtype=torch.float32, backend="cann"), (a + 1).float())
    assert select.call_count == 3
    for out in (torch.empty_like(a), torch.empty_like(a).T):
        assert mm(a, b, out=out, backend="cann") is out
        torch.testing.assert_close(out, a)
    assert select.call_count == 5


@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
def test_multiple_outputs_use_current_inputs_on_hit(npu_runtime, monkeypatch):
    import math

    from flashinfer.runtime import clear_dispatch_cache, runner

    torch, _ = npu_runtime
    select = Mock(wraps=runner.dispatch)
    monkeypatch.setattr(runner, "dispatch", select)
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    clear_dispatch_cache()
    v = torch.ones((4, 2, 64), dtype=torch.float16, device="npu")
    s = torch.zeros((4, 2), dtype=torch.float32, device=v.device)
    for value in (v, v + 4):
        output, lse = flashinfer.merge_state(value, s, value + 2, s, backend="cann")
        torch.testing.assert_close(output, value + 1)
        torch.testing.assert_close(lse, torch.full_like(s, math.log(2)), atol=1e-5, rtol=1e-5)
    assert select.call_count == 1


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
def test_second_native_operator_and_preselected_handoff(npu_runtime, aot_runtime, monkeypatch):
    from flashinfer import gemm
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.runtime import clear_dispatch_cache, runner

    torch, _ = npu_runtime
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    clear_dispatch_cache()
    select = Mock(wraps=gemm.dispatch)
    monkeypatch.setattr(gemm, "dispatch", select)
    a = torch.zeros((16, 16), dtype=torch.bfloat16, device="npu")
    out = torch.empty_like(a)
    for bias_value in (1., 3.):
        bias = torch.full_like(a[0], bias_value)
        gemm.matmul_bias_silu(a, a, bias, out=out, backend="ascendc")
        expected = torch.nn.functional.silu(torch.full((16, 16), bias_value)).to(a.dtype)
        torch.testing.assert_close(out.cpu(), expected)
    assert select.call_count == 1

    # Custom frontends can pass a kernel ID chosen by their existing dispatch.
    bind = Mock(wraps=runner.selected_runner)
    monkeypatch.setattr(runner, "selected_runner", bind)
    for bias_value in (2., 4.):
        bias = torch.full_like(a[0], bias_value)
        runner.run_selected(MATMUL_BIAS_SILU_KERNEL_ID, a, a, bias, out)
        expected = torch.nn.functional.silu(torch.full((16, 16), bias_value)).to(a.dtype)
        torch.testing.assert_close(out.cpu(), expected)
    assert bind.call_count == 1
