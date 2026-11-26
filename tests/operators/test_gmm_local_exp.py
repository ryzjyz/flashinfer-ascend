"""Local expert slicing and output preservation, beyond the full-expert matrix."""

import pytest

import flashinfer
from flashinfer.operators.gemm.gmm_local_exp import cases

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("trans_b,is_b_nz", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("promote", [False, True])
def test_local_rows_and_device_counts(npu_runtime, zero, dtype, trans_b, is_b_nz, promote):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, dtype=dtype, start=1, end=3,
                                           trans_b=trans_b, is_b_nz=is_b_nz, type_promotion=promote)
    call = flashinfer.gmm_local_exp_with_zero if zero else flashinfer.gmm_local_exp
    out = torch.full_like(expected, 123).npu()
    # group_list remains device-resident. Local expert 1 is empty, expert 2 has 9 rows.
    assert args[2].device.type == "npu"
    if not zero:
        expected[:17] = 123
        expected[26:] = 123
    returned = call(*args, **options, out=out)
    assert returned is out
    atol, rtol = cases.tolerances(dict(dtype=dtype, type_promotion=promote))[0]
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    assert torch.equal(out[:17].cpu(), expected[:17])
    assert torch.equal(out[26:].cpu(), expected[26:])


@pytest.mark.parametrize("zero", [False, True])
def test_empty_local_expert(npu_runtime, zero):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, start=1, end=2)
    out = torch.full_like(expected, 123).npu()
    call = flashinfer.gmm_local_exp_with_zero if zero else flashinfer.gmm_local_exp
    call(*args, **options, out=out)
    assert torch.equal(out.cpu(), torch.full_like(expected, 0 if zero else 123))


@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32])
def test_small_zero_output(npu_runtime, promote, n):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, counts=(1, 0, 0, 0), k=16, n=n,
                                           type_promotion=promote)
    result = flashinfer.gmm_local_exp_with_zero(*args, **options)
    atol, rtol = cases.tolerances(dict(type_promotion=promote))[0]
    torch.testing.assert_close(result.cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_real_nz_storage(npu_runtime, zero, dtype):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, dtype=dtype, trans_b=True)
    args = list(args)
    args[1] = torch_npu.npu_format_cast(args[1], 29)
    assert torch_npu.get_npu_format(args[1]) == 29
    options["is_b_nz"] = True
    call = flashinfer.gmm_local_exp_with_zero if zero else flashinfer.gmm_local_exp
    result = call(*args, **options)
    atol, rtol = cases.tolerances(dict(dtype=dtype))[0]
    torch.testing.assert_close(result.cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("n", [17, 32, 96])
def test_output_view_preserves_guard_elements(npu_runtime, zero, n):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, counts=(1, 0, 0, 0), k=16, n=n)
    storage = torch.full((32 + n + 256,), 1234, dtype=torch.float16).npu()
    out = storage[32:32 + n].reshape(1, n)
    call = flashinfer.gmm_local_exp_with_zero if zero else flashinfer.gmm_local_exp
    assert call(*args, **options, out=out) is out
    actual = storage.cpu()
    assert torch.all(actual[:32] == 1234)
    assert torch.all(actual[32 + n:] == 1234)
    torch.testing.assert_close(out.cpu(), expected, atol=0.015625, rtol=0.002)


@pytest.mark.parametrize("problem", ["dtype", "rank", "range", "weight_count", "transpose", "nz", "k"])
def test_invalid_metadata(npu_runtime, problem):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu)
    args = list(args)
    if problem == "dtype":
        args[2] = args[2].long()
    elif problem == "rank":
        args[2] = args[2].reshape(2, 2)
    elif problem == "range":
        args[3] = -1
    elif problem == "weight_count":
        args[1] = args[1][:3]
    elif problem == "transpose":
        options["trans_a"] = True
    elif problem == "nz":
        options["is_b_nz"] = True
    elif problem == "k":
        args[0] = args[0][:, :32].contiguous()
    with pytest.raises(ValueError):
        flashinfer.gmm_local_exp(*args, **options)
