"""Reviewed Ascend950 IRS registration and wheel target."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = (
    ReviewedKernel(
        "ascendc/attention/index_region_score/catlass/Ascend950",
        CapabilitySpec(
            op="attention.index_region_score",
            soc_models=("Ascend950",),
            dtypes=("float8_e4m3fn", "bfloat16", "float16", "float32", "int32", "none"),
            cases=(CapabilityCase(
                name="region_scores",
                compute_dtypes=("float8_e4m3fn", "bfloat16", "float16"),
                accumulator_dtypes=("float32",),
                kv_cache_types=("dense", "paged"),
                causal_values=(False, True),
            ),),
        ),
        backend="ascendc",
        native_recipe=("index_region_score", "catlass"),
    ),
)
