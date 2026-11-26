"""Common Matmul contract shared by every execution backend; import-safe."""

from types import SimpleNamespace
from functools import lru_cache
from typing import NamedTuple


OP = "gemm.matmul"
AOT_KERNEL_ID = "ascendc/gemm/matmul/native/op_package/Ascend950"
CANN_KERNEL_ID = "cann/gemm/matmul_v1"
TRITON_KERNEL_ID = "triton/gemm/matmul_v1/Ascend950"


def contract():
    from .ascendc import load_contract

    return load_contract(AOT_KERNEL_ID)


def output_metadata(a, b, out=None):
    if out is not None:
        return out
    return SimpleNamespace(
        shape=(a.shape[0] if len(a.shape) else 0,
               b.shape[1] if len(b.shape) >= 2 else 0),
        dtype=a.dtype, device=a.device, is_contiguous=lambda: True,
    )


def metadata_reason(a, b, out):
    a_shape, b_shape, out_shape = a.shape, b.shape, out.shape
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        return "Matmul requires rank-2 tensors"
    m, k = a_shape
    n = b_shape[1]
    if k != b_shape[0] or tuple(out_shape) != (m, n):
        return "Matmul input/output dimensions are inconsistent"
    if min(m, n, k) <= 0:
        return "Matmul dimensions must be positive"
    if a.dtype != b.dtype or a.dtype != out.dtype or a.device != b.device or a.device != out.device:
        return "Matmul tensors must share dtype and device"
    return "supported"


class _ParameterRole(NamedTuple):
    role: str


@lru_cache(maxsize=8)
def overlap_parameters(count=3):
    return tuple(_ParameterRole("output" if i == count - 1 else "input") for i in range(count))


def validate_tensors(a, b, out):
    """Shared execution semantics, independent of any frozen AOT recipe."""
    import torch
    import torch_npu
    from .jit.tensor_adapter import _reject_output_overlap

    reason = metadata_reason(a, b, out)
    if reason != "supported":
        raise ValueError(reason)
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError("Matmul requires NPU tensors")
    if any(not t.is_contiguous() or torch_npu.get_npu_format(t) != 2 for t in (a, b, out)):
        raise ValueError("Matmul requires contiguous ND tensors")
    _reject_output_overlap(overlap_parameters(), (a, b, out))
