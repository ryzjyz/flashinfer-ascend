"""Ascend950 dense prefill using the CATLASS DSL JIT compiler."""

from flashinfer.operators.attention.single_prefill_catlass.operator import OPERATOR
from flashinfer.operators.author_api import CatlassDslImplementation, DimRange, PythonKernel, Support
from flashinfer.runtime import KernelConstraints

from .runner import CatlassDslPrefillRunner


IMPLEMENTATION = CatlassDslImplementation(
    operator=OPERATOR,
    kernel_id="catlass_dsl/attention/single_prefill_catlass/Ascend950",
    target_soc="Ascend950",
    support=Support(
        dimensions={
            "S": DimRange(128, 512, 128),
            "T": DimRange(128, 512, 128),
            "H": DimRange(1, 8),
            "D": DimRange(128, 128),
        },
        dtypes=("float16", "bfloat16"),
        soc_models=("Ascend950",),
        physical_layouts=("ND",),
        specializations=(("tensor_contract", ("supported",)),),
        contiguous=True,
    ),
    constraints=KernelConstraints(dtype_signatures=(("float16",) * 4, ("bfloat16",) * 4)),
    source=PythonKernel(language="catlass_dsl"),
    runner_factory=CatlassDslPrefillRunner,
)
