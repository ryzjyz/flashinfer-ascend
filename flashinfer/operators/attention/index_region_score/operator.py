"""Weighted head scores and request-local region selection."""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.index_region_score"
INPUT_NAMES = ("query", "weights", "key_summary", "actual_kv_lengths", "block_table", "cu_seq_lens")
ATTRIBUTE_NAMES = ("mask_mode", "head_scale", "cache_mode", "max_region_len", "region_size", "top_k", "sort_mode", "return_score")


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def infer(arguments, layouts):
    q, weights, key, lengths, table, cu = (arguments[name] for name in INPUT_NAMES)
    attrs = {name: arguments[name] for name in ATTRIBUTE_NAMES}
    for name in ("mask_mode", "cache_mode", "max_region_len", "region_size", "top_k", "sort_mode"):
        if type(attrs[name]) is not int:
            raise ValueError(name + " must be an integer")
    if attrs["mask_mode"] not in (0, 3) or attrs["cache_mode"] not in (0, 1):
        raise ValueError("mask_mode must be 0 or 3 and cache_mode must be 0 or 1")
    if not 1 <= attrs["top_k"] <= 4096:
        raise ValueError("top_k must be in [1,4096]")
    if attrs["sort_mode"] != 0:
        raise ValueError("sort_mode must be 0 (ascending region ID)")
    if not 1 <= attrs["region_size"] <= 2**31 - 1:
        raise ValueError("region_size must be in [1,2147483647]")
    scale = attrs["head_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("head_scale must be finite and representable as float32")
    if type(attrs["return_score"]) is not bool:
        raise ValueError("return_score must be bool")
    if len(q.shape) != 4 or any(int(d) <= 0 for d in q.shape):
        raise ValueError("query requires positive [T,G,H,D] dimensions")
    tokens, groups, heads, dim = q.shape
    if tokens > 2**31 - 1 or groups > 2**32 - 1 or (groups != 1 and groups % 4):
        raise ValueError("T must fit int32 and G must be 1 or a multiple of 4 fitting uint32")
    if not 4 <= heads <= 128 or heads % 4 or dim not in (128, 256):
        raise ValueError("H must be a multiple of 4 in [4,128] and D must be 128 or 256")
    dtype = _dtype(q)
    if dtype not in ("float8_e4m3fn", "float16", "bfloat16") or _dtype(key) != dtype:
        raise ValueError("query and key_summary must share float8_e4m3fn, float16 or bfloat16 dtype")
    if dim == 128 and dtype == "float8_e4m3fn":
        raise ValueError("D128 requires float16 or bfloat16 inputs")
    if tuple(weights.shape) != (tokens, groups, heads) or _dtype(weights) not in ("float16", "bfloat16", "float32"):
        raise ValueError("weights requires [T,G,H] and float16, bfloat16 or float32 dtype")
    paged = attrs["cache_mode"] == 1
    if paged:
        if len(key.shape) != 4 or key.shape[2:] != (1, dim) or any(int(d) <= 0 for d in key.shape):
            raise ValueError("paged key_summary requires positive [P,S,1,D] dimensions")
        if key.shape[1] > 1024:
            raise ValueError("paged S must be in [1,1024]")
        if table is None or len(table.shape) != 2 or min(table.shape) <= 0 or _dtype(table) != "int32":
            raise ValueError("paged mode requires int32 block_table[N,C]")
        requests = table.shape[0]
        if cu is None or tuple(cu.shape) != (requests + 1,) or _dtype(cu) != "int32":
            raise ValueError("paged mode requires int32 cu_seq_lens[N+1]")
        regions = attrs["max_region_len"]
        if not 0 <= regions <= min(table.shape[1] * key.shape[1], 2**31 - 1):
            raise ValueError("paged max_region_len must fit block-table capacity and int32")
    else:
        if tokens > 2**20 or groups > 4096:
            raise ValueError("continuous T must be in [1,1048576] and G must be at most 4096")
        if len(key.shape) != 3 or key.shape[1:] != (1, dim) or not 0 < key.shape[0] <= 2**31 - 1:
            raise ValueError("continuous key_summary requires [R,1,D] with positive int32 R")
        if table is not None:
            raise ValueError("continuous mode does not accept block_table")
        regions = key.shape[0] if attrs["max_region_len"] == -1 else attrs["max_region_len"]
        if attrs["max_region_len"] < -1 or not 0 <= regions <= key.shape[0]:
            raise ValueError("continuous max_region_len must be -1 or in [0,R]")
        requests = 1
        if cu is not None:
            if attrs["mask_mode"] != 3 or len(cu.shape) != 1 or cu.shape[0] < 2 or _dtype(cu) != "int32":
                raise ValueError("continuous cu_seq_lens requires causal mode and int32 [N+1]")
            requests = cu.shape[0] - 1
    if lengths is not None and (attrs["mask_mode"] != 3 or tuple(lengths.shape) != (requests,) or _dtype(lengths) != "int32"):
        raise ValueError("actual_kv_lengths requires causal mode and int32 [N]")
    if attrs["mask_mode"] == 3 and lengths is None:
        raise ValueError("causal mode requires actual_kv_lengths")
    if regions > 16777216 or tokens * groups > 2**32 - 1:
        raise ValueError("dimensions exceed selector address limits")
    score_shape = (groups, tokens, regions) if attrs["return_score"] else (0,)
    supplied = arguments.get("out")
    if attrs["return_score"] and isinstance(supplied, (tuple, list)) and len(supplied) == 2:
        shape = tuple(supplied[0].shape)
        if len(shape) == 3 and shape[:2] == (groups, tokens) and regions <= shape[2] <= 2**32 - 1:
            score_shape = shape
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (OutputSpec(score_shape, "float32", q.device),
         OutputSpec((groups, tokens, attrs["top_k"]), "int32", q.device)),
        attrs,
        InferenceSpec(compute_dtype=dtype, accumulator_dtype="float32", causal=attrs["mask_mode"] == 3,
                      kv_cache_type="paged" if paged else "dense"),
    )


OPERATOR = OperatorDefinition(
    name="index_region_score", domain="attention", op=OP,
    schema="(Tensor query, Tensor weights, Tensor key_summary, *, Tensor? actual_kv_lengths=None, Tensor? block_table=None, Tensor? cu_seq_lens=None, int mask_mode=0, float head_scale=1.0, int cache_mode=0, int max_region_len=-1, int region_size=8, int top_k=512, int sort_mode=0, bool return_score=True) -> (Tensor, Tensor)",
    output_names=("score", "selected_region_ids"), infer=infer,
    tensor_shapes=(("T", "G", "H", "D"), ("T", "G", "H")),
    description=(
        "Sum weighted ReLU query-key products over heads, then select request-local regions. "
        "The only supported sort_mode is 0, which returns selected IDs in ascending region ID order. "
        "IDs are padded with -1. Score values are defined only for visible regions "
        "within the logical region extent. Values outside that range are unspecified. "
        "return_score=False returns an empty score Tensor and bounded-workspace IDs."
    ),
)
