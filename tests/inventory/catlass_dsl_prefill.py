"""Reviewed hardware and dense prefill limits for the CATLASS DSL backend."""

from flashinfer.runtime import CapabilitySpec, InputSignature
from tests.inventory import ReviewedKernel, shape_ranges


KERNELS = (
    ReviewedKernel(
        "catlass_dsl/attention/single_prefill_catlass/Ascend950",
        CapabilitySpec(
            op="ascendc.single_prefill_catlass",
            dtypes=("float16", "bfloat16"),
            soc_models=("Ascend950",),
            physical_layouts=("ND",),
            specializations=(("tensor_contract", ("supported",)),),
            shape_ranges=shape_ranges(
                (("S", "H", "D"), ("T", "H", "D"), ("T", "H", "D"), ("S", "H", "D")),
                {"S": (128, 512), "T": (128, 512), "H": (1, 8), "D": (128, 128)},
            ),
        ),
        backend="catlass_dsl",
        source=("catlass_dsl", "Ascend950"),
        smoke_signature=InputSignature(
            "ascendc.single_prefill_catlass",
            dtypes=("float16",) * 4,
            shapes=((128, 2, 128), (256, 2, 128), (256, 2, 128), (128, 2, 128)),
            physical_layouts=("ND",) * 4,
            specializations=(("tensor_contract", "supported"),),
        ),
    ),
)
