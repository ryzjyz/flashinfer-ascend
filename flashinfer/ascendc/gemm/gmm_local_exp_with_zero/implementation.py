"""Independent arch22 native binding for GmmLocalExpWithZero."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import TensorParameter
from flashinfer.operators.author_api import DimRange, Support
from flashinfer.operators.gemm.gmm_local_exp_with_zero.operator import OPERATOR
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS

DTYPES = ("float16", "bfloat16")
SIGNATURES = tuple((d, d, "int32", out) for d in DTYPES for out in (d, "float32"))

IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR, default_target="Ascend910B",
    support=Support(soc=ARCH22_SOCS, layout=("ND", "NZ"), contiguous=True,
                    cases=(CapabilityCase(name="local_expert", dtype_signatures=SIGNATURES,
                                          compute_dtypes=DTYPES, accumulator_dtypes=("float32",)),),
                    dimensions={n: DimRange(1, 2**31 - 1) for n in ("M", "K", "E", "W0", "W1", "G", "N")}),
    build=BuildConfig(device="op_kernel/gmm_local_exp_with_zero.cpp", host=("op_host/gmm_local_exp_with_zero_def.cpp",),
                      op_type="GmmLocalExpWithZero", tiling="TilingGmmLocalExpWithZero", device_entry="gmm_local_exp_with_zero",
                      compile_options=("-Wno-deprecated-declarations",)),
    binding=NativeBinding(
        tensors=(TensorParameter("x", "input", "float16", ("M", "K")),
                 TensorParameter("weight", "input", "float16", ("E", "W0", "W1"), physical_layouts=("ND", "NZ")),
                 TensorParameter("group_list", "input", "int32", ("G",)),
                 TensorParameter("y", "output", "float16", ("M", "N"))),
        dtype_signatures=SIGNATURES,
        attributes=(AttributeParameter("trans_b", "bool", default=False),
                    AttributeParameter("is_b_nz", "bool", default=False),
                    AttributeParameter("exp_start_idx", "int64", default=0, minimum=0, maximum=2**31 - 1),
                    AttributeParameter("exp_end_idx", "int64", default=1, minimum=1, maximum=2**31 - 1))),
    workspace_bound=16 * 1024 * 1024,
)
