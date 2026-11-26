"""Reviewed A3 DS4.1 MLA prolog capability and op-package target."""

from flashinfer.operators.attention.mla_prolog.operator import FUSION_OPS, QUANTIZATION
from flashinfer.runtime import CapabilityCase, CapabilitySpec, FeatureConstraint, ShapeRange
from tests.inventory import ReviewedKernel


DTYPE_SIGNATURE = (
    "bfloat16", "bfloat16", "bfloat16", "int8", "bfloat16", "bfloat16",
    "bfloat16", "float32", "float32", "int32", "bfloat16", "bfloat16",
    "bfloat16",
)

KERNELS = (
    ReviewedKernel(
        "ascendc/attention/mla_prolog/catlass/op_package/Ascend910C",
        CapabilitySpec(
            op="attention.mla_prolog",
            soc_models=("Ascend910C",),
            dtypes=("bfloat16", "int8", "float32", "int32"),
            physical_layouts=("ND", "NZ"),
            shape_ranges=tuple(
                ShapeRange("T", tensor_index, 0, 1, 4096)
                for tensor_index in (0, 7, 8, 9, 12)
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
