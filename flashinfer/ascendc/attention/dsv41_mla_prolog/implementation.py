"""Ascend C declaration for attention.dsv41_mla_prolog.

SKELETON, NOT A VERIFIED IMPLEMENTATION. The public tensor/attribute contract
below is taken from the standalone kernel host header (``Dsv41MlaPrologArguments``)
and the frozen definition. The native launch path is still unresolved: the
current op_kernel is a CATLASS ``DeviceGemm`` host launcher that allocates its
own workspace and calls ``aclrtGetHardwareSyncAddr``; it does not define the
framework-required ``__global__ __aicore__ fi_<op>`` entry or consume a tiling
struct. Do not build this target or claim support until the ABI is agreed with
the maintainers. See the submission gap analysis.
"""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    BuildConfig,
    NativeBinding,
)
from flashinfer.operators.attention.dsv41_mla_prolog.operator import (
    FUSION_OPS,
    OPERATOR,
    QUANTIZATION,
)
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase, FeatureConstraint, ShapeRange
from flashinfer.jit.recipe import (
    BuildDependency,
    CATLASS_REVISION,
    ShapeConstraint,
    TensorParameter,
)
from .runner import make_runner

# TODO(inventory): the final kernel_id/implementation segment ("native" here
# versus the CATLASS build in BuildConfig below) must be agreed with maintainers
# together with tests/production_inventory.py. It is deliberately left at the
# id already generated for this local skeleton.
KERNEL_ID = "ascendc/attention/dsv41_mla_prolog/native/Ascend910C"
OP = "attention.dsv41_mla_prolog"

# Inputs followed by the one fresh output, in NativeBinding order.
DTYPE_SIGNATURE = (
    "bfloat16", "bfloat16", "bfloat16", "int8", "bfloat16", "bfloat16",
    "bfloat16", "float32", "float32", "int32", "bfloat16", "bfloat16",
)


def _tensor(name, role, dtype, shape, *, physical_layouts=("ND",)):
    return TensorParameter(
        name=name, role=role, dtype=dtype, shape=shape,
        physical_layouts=physical_layouts,
    )


IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    support=Support(
        soc_models=("Ascend910C",),
        dtypes=("bfloat16", "int8", "float32", "int32"),
        physical_layouts=("ND", "NZ"),
        shape_ranges=(
            ShapeRange(name="T", tensor_index=0, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=7, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=8, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=9, axis=0, minimum=1, maximum=4096),
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
    build=BuildConfig(
        device="op_kernel/dsv41_mla_prolog.cpp",
        host=("op_host/dsv41_mla_prolog_def.cpp",),
        op_type="FiDsv41MlaProlog",
        tiling="TilingDsv41MlaProlog",
        includes=("op_kernel", "op_host"),
        cmake="CMakeLists.txt",
        implementation="catlass",
        dependencies=(
            BuildDependency(name="catlass", revision=CATLASS_REVISION),
        ),
        definitions=("CATLASS_ARCH=2201",),
        compile_options=("-O2",),
        link_options=("-lascendcl",),
        workspace_bytes="dynamic",
        # TODO(ABI): the fields below are intentionally left to the framework
        # defaults so this file does not invent symbols. The defaults are
        # fi_dsv41_mla_prolog / aclnnFiDsv41MlaProlog / aclnn.dsv41_mla_prolog.v2.
        # None of them exists in the current CATLASS host-launched kernel.
        device_entry=None,
        host_entry=None,
        launch_abi=None,
        package_vendor="flashinfer_dsv41_mla_prolog",
        recipe_id="dsv41_mla_prolog",
    ),
    binding=NativeBinding(
        tensors=(
            _tensor("x", "input", "bfloat16", ("T", 5120)),
            _tensor("w_qa", "input", "bfloat16", (1280, 5120)),
            _tensor("gamma_qa", "input", "bfloat16", (1280,)),
            _tensor("w_qb", "input", "int8", (1024, 80, 16, 32),
                    physical_layouts=("NZ",)),
            _tensor("w_qb_scale", "input", "bfloat16", (32768,)),
            _tensor("w_kv", "input", "bfloat16", (512, 5120)),
            _tensor("gamma_kv", "input", "bfloat16", (512,)),
            _tensor("rope_cos", "input", "float32", ("T", 1, 1, 64)),
            _tensor("rope_sin", "input", "float32", ("T", 1, 1, 64)),
            _tensor("slot_indices", "input", "int32", ("T", 2)),
            # The kernel writes the addressed rows of this input in place.
            _tensor("kv_cache", "input", "bfloat16", (22594, 128, 512)),
            _tensor("q_output", "output", "bfloat16", ("T", 64, 512)),
        ),
        dimensions=(ShapeConstraint(name="T", minimum=1, maximum=4096),),
        dtype_signatures=(DTYPE_SIGNATURE,),
        attributes=(),
    ),
    runner_factory=make_runner,
    # Conservative bound for T=4096 and 24 AIC, derived from the kernel's
    # MakeWorkspaceLayout (about 99 MiB); the exact size is host-computed.
    workspace_bound=128 * 1024 * 1024,
    allow_jit_fallback=False,
)

IMPLEMENTATIONS = (IMPLEMENTATION,)
NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
