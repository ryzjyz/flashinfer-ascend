"""Reviewed arch22 DSA sparse prolog capabilities and wheel targets."""

from flashinfer.runtime import CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/dsa_sparse_prolog/native/" + soc,
        CapabilitySpec(
            op="attention.dsa_sparse_prolog",
            soc_models=(soc,),
            dtypes=("float16", "bfloat16", "float32"),
            physical_layouts=("ND", "NZ"),
            specializations=(("tensor_contract", ("supported",)),),
        ),
        backend="ascendc",
        native_recipe=("dsa_sparse_prolog", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
