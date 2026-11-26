"""Reviewed per-expert-count GEMM dtype and physical target contracts."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel, shape_ranges

AXES = (("M", "K"), ("E", "W0", "W1"), ("G",), ("M", "N"))
BOUNDS = {n: (1, 2**31 - 1) for shape in AXES for n in shape}
KERNELS = tuple(
    ReviewedKernel(
        "ascendc/gemm/gmm_local_exp/native/" + soc,
        CapabilitySpec(op="gemm.gmm_local_exp", soc_models=(soc,), physical_layouts=("ND", "NZ"),
                       cases=(CapabilityCase(name="local_expert", compute_dtypes=("float16", "bfloat16"),
                                             accumulator_dtypes=("float32",),
                                             dtype_signatures=(("float16", "float16", "int32", "float16"),
                                                               ("float16", "float16", "int32", "float32"),
                                                               ("bfloat16", "bfloat16", "int32", "bfloat16"),
                                                               ("bfloat16", "bfloat16", "int32", "float32"))),),
                       shape_ranges=shape_ranges(AXES, BOUNDS)),
        backend="ascendc", native_recipe=("gmm_local_exp", "native"),
    ) for soc in ("Ascend910B", "Ascend910C")
)
