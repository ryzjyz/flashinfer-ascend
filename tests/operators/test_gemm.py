"""Public GEMM numerical comparisons; setup and compilation live in fixtures."""

import pytest

import flashinfer
from flashinfer.operators.gemm.matmul import cases

pytestmark = pytest.mark.requires_soc("Ascend950")


@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 48), (80, 48, 32),
                                  (128, 128, 128), (256, 256, 256), (256, 16, 256), (16, 256, 16)])
def test_matmul_accuracy(npu_runtime, matmul_backend, shape):
    torch, torch_npu = npu_runtime
    m, n, k = shape
    args, options, expected = cases.sample(torch, torch_npu, m=m, k=k, n=n)
    actual = flashinfer.matmul(*args, **options, backend=matmul_backend)
    (atol, rtol), = cases.TOLERANCES
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)


@pytest.mark.real_compile
def test_warm_native_matmul_rechecks_shape_storage_and_views(npu_runtime, aot_runtime):
    from flashinfer._matmul import AOT_KERNEL_ID
    from flashinfer.runtime import SchemaError, get_global_registry

    torch, torch_npu = npu_runtime
    (a, b), _, expected = cases.sample(torch, torch_npu, m=16, k=16, n=16)
    out = torch.empty_like(a)
    flashinfer.matmul(a, b, out=out, kernel_id=AOT_KERNEL_ID)
    runner = get_global_registry().get(AOT_KERNEL_ID).runner.resolve()
    with pytest.raises(ValueError, match="overlap"):
        runner(a, b, a.view_as(out))
    with pytest.raises(ValueError, match="contiguous"):
        runner(a, b, out.t())
    with pytest.raises(SchemaError):
        runner(a, b, out[:, :8])
    with pytest.raises(SchemaError):
        runner(a, b, out.float())
    runner(a.clone(), b, out)
    (atol, rtol), = cases.TOLERANCES
    torch.testing.assert_close(out.cpu().double(), expected, atol=atol, rtol=rtol)
