"""Public semantics of GmmLocalExpWithZero; group_list contains per-expert counts."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec


def infer(arguments, layouts):
    x, weight, groups = (arguments[n] for n in ("x", "weight", "group_list"))
    start, end = (arguments[n] for n in ("exp_start_idx", "exp_end_idx"))
    trans_b, is_b_nz = (arguments[n] for n in ("trans_b", "is_b_nz"))
    if arguments["trans_a"]:
        raise ValueError("trans_a=True is not supported")
    if x.ndim != 2 or weight.ndim != 3 or groups.ndim != 1:
        raise ValueError("x, weight and group_list require ranks 2, 3 and 1")
    dtype = str(x.dtype).removeprefix("torch.")
    if dtype not in ("float16", "bfloat16") or weight.dtype != x.dtype:
        raise ValueError("x and weight must share float16 or bfloat16 dtype")
    if str(groups.dtype) != "torch.int32":
        raise ValueError("group_list must have native int32 dtype")
    if not 0 <= start < end <= groups.shape[0] or weight.shape[0] != end - start:
        raise ValueError("weight must contain exactly the experts in [exp_start_idx, exp_end_idx)")
    m, k = x.shape
    wk, n = (weight.shape[2], weight.shape[1]) if trans_b else weight.shape[1:]
    if min(m, k, n) <= 0 or wk != k:
        raise ValueError("positive matrix dimensions and matching K are required")
    if k > 65535 or weight.shape[2] > 65535:
        raise ValueError("native matmul strides must not exceed 65535")
    if max(m * k, m * n, weight.numel()) > 2**31 - 1:
        raise ValueError("matrix element offsets exceed the native index range")
    if is_b_nz and (not trans_b or k % 16 or n % 16):
        raise ValueError("NZ requires trans_b=True and K/N multiples of 16")
    if layouts is not None and layouts["weight"] == "NZ" and not is_b_nz:
        raise ValueError("NZ storage requires is_b_nz=True")
    output_dtype = "float32" if arguments["type_promotion"] else dtype
    if m * n * (4 if output_dtype == "float32" else 2) > 2**32 - 1:
        raise ValueError("output byte count exceeds the native index range")
    attributes = dict(trans_b=trans_b, is_b_nz=is_b_nz, exp_start_idx=start, exp_end_idx=end)
    return Invocation((x, weight, groups), (OutputSpec((m, n), output_dtype, x.device),),
                      attributes, InferenceSpec(compute_dtype=dtype, accumulator_dtype="float32"))


OPERATOR = OperatorDefinition(
    name="gmm_local_exp_with_zero", domain="gemm", op="gemm.gmm_local_exp_with_zero",
    schema="(Tensor x, Tensor weight, Tensor group_list, int exp_start_idx, int exp_end_idx, "
           "*, bool trans_a=False, bool trans_b=False, bool is_b_nz=False, bool type_promotion=False) -> Tensor",
    output_names=("y",), infer=infer,
    tensor_shapes=(("M", "K"), ("E", "W0", "W1"), ("G",), ("M", "N")),
    description="GmmLocalExpWithZero: local-expert grouped matmul with per-expert token counts. "
                "Nonlocal output rows are zeroed.",
)
