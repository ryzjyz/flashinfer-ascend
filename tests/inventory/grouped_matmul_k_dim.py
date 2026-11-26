"""Reviewed K-axis grouped matmul capability on arch22 (Ascend910B/910C).

Ported from cann/ops-transformer MR !12993. Groups share the K axis and each
emit one [M, N] tile, giving a [G, M, N] output; group_list is an int64
cumulative prefix. Four dtype signatures (2 compute x 2 output) per SoC.
"""

from flashinfer.runtime import (
    CapabilityCase, CapabilitySpec, InferenceSpec, InputSignature, ShapeRange,
)
from tests.inventory import ReviewedKernel


CASES = tuple(
    CapabilityCase(
        name="%s_to_%s" % (compute, output),
        dtype_signatures=((compute, compute, "int64", output),),
        compute_dtypes=(compute,),
        accumulator_dtypes=("float32",),
        shape_ranges=(ShapeRange("XB", 0, 1, 1, 65535), ShapeRange("N", 1, 1, 1, 65535)),
        specializations=(("tensor_contract", ("supported",)),),
    )
    for compute in ("float16", "bfloat16")
    for output in (compute, "float32")
)

# Three operands plus a prospective output and an explicit compute contract, so
# the generic two-tensor dummy signature cannot reach this operator.
SMOKE_SIGNATURE = InputSignature(
    "gemm.grouped_matmul_k_dim",
    dtypes=("float16", "float16", "int64", "float16"),
    shapes=((64, 96), (96, 96), (4,), (4, 64, 96)),
    physical_layouts=("ND",) * 4,
    specializations=(("tensor_contract", "supported"),),
    inference=InferenceSpec(compute_dtype="float16", accumulator_dtype="float32"),
)

KERNELS = tuple(
    ReviewedKernel(
        "ascendc/gemm/grouped_matmul_k_dim/native/op_package/" + soc,
        CapabilitySpec(
            op="gemm.grouped_matmul_k_dim",
            soc_models=(soc,),
            physical_layouts=("ND",),
            cases=CASES,
        ),
        backend="ascendc",
        native_recipe=("grouped_matmul_k_dim", "native"),
        smoke_signature=SMOKE_SIGNATURE,
    )
    for soc in ("Ascend910B", "Ascend910C")
) + (
    # Device-agnostic plain-torch reference; no SoC or layout claim, same cases.
    ReviewedKernel(
        "reference/gemm/grouped_matmul_k_dim_v1",
        CapabilitySpec(op="gemm.grouped_matmul_k_dim", cases=CASES),
        backend="reference",
        smoke_signature=SMOKE_SIGNATURE,
    ),
)
