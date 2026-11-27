"""Native Ascend950 package declaration for Index Region Score."""

from dataclasses import replace
import hashlib
from pathlib import Path

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import BuildDependency, CATLASS_REVISION, ShapeConstraint, TensorParameter
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase, SchemaError, SourceRef

from flashinfer.operators.attention.index_region_score.operator import OPERATOR
from .runner import admission, make_runner

QUERY_DTYPES = ("float8_e4m3fn", "bfloat16", "float16")
WEIGHT_DTYPES = ("float16", "bfloat16", "float32")
KERNEL_ID = "ascendc/attention/index_region_score/catlass/Ascend950"

_COMMON_INCLUDE = Path("csrc/kernels/sparse_indexer/common/include")
_COMMON_HEADERS = (
    "block/block_schedule.hpp",
    "contracts/common.hpp",
    "primitive/topk_fp32.hpp",
    "sync/subblock_barrier.hpp",
)


class _IndexRegionScoreImplementation(AscendCImplementation):
    def prepare_outputs(self, root):
        prepared = super().prepare_outputs(root)
        root = Path(root).resolve()
        headers = []
        for name in _COMMON_HEADERS:
            relative = _COMMON_INCLUDE / name
            header = root / relative
            if not header.is_file() or header.resolve() != header:
                raise SchemaError(f"IRS shared build input must be a regular file: {relative}")
            headers.append(SourceRef("ascendc", relative.as_posix(), hashlib.sha256(header.read_bytes()).hexdigest()))

        recipe = prepared["recipe"]
        recipe = replace(recipe, headers=recipe.headers + tuple(headers),
                         include_dirs=recipe.include_dirs + (_COMMON_INCLUDE.as_posix(),))
        prepared["recipe"] = recipe
        prepared["outputs"] = {path: [recipe.to_dict()] for path in prepared["outputs"]}
        return prepared


IMPLEMENTATION = _IndexRegionScoreImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    name="catlass",
    source_dir="csrc/kernels/sparse_indexer/index_region_score",
    support=Support(
        soc_models=("Ascend950",),
        dtypes=QUERY_DTYPES + ("float32", "int32", "none"),
        cases=(CapabilityCase(
            name="region_scores",
            compute_dtypes=QUERY_DTYPES,
            accumulator_dtypes=("float32",),
            kv_cache_types=("dense", "paged"),
            causal_values=(False, True),
        ),),
    ),
    build=BuildConfig(
        device="op_kernel/indexer_region_score.cpp",
        host=("op_host/index_region_score.cpp", "op_host/indexer_region_score_tiling.cpp"),
        op_type="FiIndexerRegionScore",
        device_entry="fi_indexer_region_score",
        tiling="FiIndexerRegionScoreTiling",
        includes=("op_kernel", "op_host", "include"),
        implementation="catlass",
        dependencies=(BuildDependency("catlass", CATLASS_REVISION),),
        definitions=("CATLASS_ARCH=3510",),
        compile_options=("-O3",),
        cmake="CMakeLists.txt",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter("query", "input", "float16", ("T", "G", "H", "D")),
            TensorParameter("weights", "input", "float16", ("T", "G", "H")),
            TensorParameter("key_summary", "input", "float16", ("P", "S", 1, "D"),
                            shape_alternatives=(("R", 1, "D"), ("KEY_SPAN",))),
            TensorParameter("actual_kv_lengths", "input", "int32", ("N",), optional=True),
            TensorParameter("block_table", "input", "int32", ("N", "C"), optional=True),
            TensorParameter("cu_seq_lens", "input", "int32", ("N_PLUS_ONE",), optional=True),
            TensorParameter("output", "output", "float32", ("G", "T", "R_OUT"),
                            shape_alternatives=((1,),)),
            TensorParameter("selected_region_ids", "output", "int32", ("G", "T", "K")),
        ),
        dimensions=tuple(
            ShapeConstraint(name, minimum, maximum, multiple)
            for name, minimum, maximum, multiple in (
                ("T", 1, 2**31 - 1, 1), ("G", 1, 2**32 - 1, 1), ("H", 4, 128, 4),
                ("D", 128, 256, 128), ("R", 1, 2**31 - 1, 1), ("R_OUT", 1, 2**32 - 1, 1),
                ("P", 1, 2**32 - 1, 1), ("S", 1, 1024, 1), ("N", 1, 2**32 - 1, 1),
                ("C", 1, 2**32 - 1, 1), ("N_PLUS_ONE", 2, 2**32, 1),
                ("KEY_SPAN", 1, 2**63 - 1, 1), ("K", 1, 4096, 1),
            )
        ),
        dtype_signatures=tuple(
            (dtype, weights, dtype, "int32", "int32", "int32", "float32", "int32")
            for dtype in QUERY_DTYPES for weights in WEIGHT_DTYPES
        ),
        attributes=(
            AttributeParameter("mask_mode", "int64", 0, 0, 3),
            AttributeParameter("head_scale", "float64", 1.0, -3.4028234663852886e38, 3.4028234663852886e38),
            AttributeParameter("cache_mode", "int64", 0, 0, 1),
            AttributeParameter("max_region_len", "int64", -1, -1, 2**31 - 1),
            AttributeParameter("region_size", "int64", 8, 1, 2**31 - 1),
            AttributeParameter("top_k", "int64", 512, 1, 4096),
            AttributeParameter("sort_mode", "int64", 0, 0, 0),
            AttributeParameter("return_score", "bool", True),
            AttributeParameter("graph_key_pages", "int64", 0, 0),
            AttributeParameter("graph_key_summaries", "int64", 0, 0, 1024),
            AttributeParameter("graph_key_page_stride", "int64", 0, 0),
        ),
    ),
    runner_factory=make_runner,
    admission=admission,
    # Covers the 256 MiB bounded score pool and the supported full-score tiling scratch.
    workspace_bound=256 * 1024 * 1024,
)

NATIVE = IMPLEMENTATION.native
