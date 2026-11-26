"""Reviewed A3 DS4.1 MLA prolog capability and op-package target."""

from flashinfer.operators.attention.mla_prolog.operator import (
    FUSION_OPS,
    HEAD_DIM,
    MAX_HEAD_NUM,
    MIN_HEAD_NUM,
    QUANTIZATION,
)
from flashinfer.runtime import CapabilityCase, CapabilitySpec, FeatureConstraint, ShapeRange
from tests.inventory import ReviewedKernel


# Eleven inputs followed by the three fresh outputs (q_output, qr, kv), in
# NativeBinding order.
DTYPE_SIGNATURE = (
    "bfloat16", "bfloat16", "bfloat16", "int8", "bfloat16", "bfloat16",
    "bfloat16", "float32", "float32", "int32", "bfloat16", "bfloat16",
    "bfloat16", "bfloat16",
)

KERNELS = (
    ReviewedKernel(
        "ascendc/attention/mla_prolog/catlass/op_package/Ascend910C",
        CapabilitySpec(
            op="attention.mla_prolog",
            soc_models=("Ascend910C",),
            dtypes=("bfloat16", "int8", "float32", "int32"),
            physical_layouts=("ND", "NZ"),
            shape_ranges=(
                *(ShapeRange("T", tensor_index, 0, 1, 4096)
                  for tensor_index in (0, 7, 8, 9, 11, 12, 13)),
                # kv is [T,512] in both modes; the fused path leaves it
                # unwritten. Local Q head count for TP sharding.
                ShapeRange("q_head_num", 11, 1, MIN_HEAD_NUM, MAX_HEAD_NUM),
                # w_qb is [q_head_num*16, 80, 16, 32] NZ and w_qb_scale is
                # [q_head_num*512]; the exact product relationship is enforced
                # by infer() and the host tiler.
                ShapeRange("q_head_num*16", 3, 0,
                           MIN_HEAD_NUM * 16, MAX_HEAD_NUM * 16),
                ShapeRange("q_head_num*512", 4, 0,
                           MIN_HEAD_NUM * HEAD_DIM, MAX_HEAD_NUM * HEAD_DIM),
            ),
            cases=(CapabilityCase(
                name="a3_bf16_fixed_chain",
                dtype_signatures=(DTYPE_SIGNATURE,),
                compute_dtypes=("bfloat16",),
                accumulator_dtypes=("float32",),
                quantization=QUANTIZATION,
                fusion_ops=FUSION_OPS,
                features=(
                    FeatureConstraint("normalization.kind", ("rms",)),
                    FeatureConstraint("normalization.affine", (True,)),
                ),
            ),),
        ),
        backend="ascendc", native_recipe=("mla_prolog", "catlass"),
    ),
)
