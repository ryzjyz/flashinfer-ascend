"""DS4.1 MLA prolog: fixed-shape fused projection, norm, quantization and RoPE.

The public contract is one BF16 token block ``x`` of shape ``[T, 5120]`` with
``T`` in ``[1, 4096]``. The fused chain is:

1. ``x @ w_qa.T`` then RMSNorm with ``gamma_qa``;
2. dynamic per-token INT8 quantization of the normalized Q-LoRA activation;
3. INT8 ``@ w_qb`` with per-token and per-output-channel dequantization;
4. partial RoPE on the last 64 of 512 head channels, producing
   ``q_output[T, 64, 512]``;
5. ``x @ w_kv.T`` then RMSNorm with ``gamma_kv``, partial RoPE, and an in-place
   scatter of the 512 values into ``kv_cache`` rows addressed by
   ``slot_indices[T, 2]``.

The cache rows not addressed by ``slot_indices`` are untouched sentinels.

IMPORTANT: ``kv_cache`` is an input Tensor that the device kernel updates in
place. The generated functional/out schema path rejects mutation of an input
that is not declared as a mutated argument. This module deliberately keeps the
metadata hook complete, but the final public entry (``public_entry`` plus
``torch_declarations`` with ``mutates_args=("kv_cache",)``, or an explicit
maintainer-approved schema extension) is still unresolved. See
``post_final/flashinfer_submission/gap_analysis.md`` in the evolution output
for the open items; do not treat this declaration as a verified public API.
"""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec, QuantSpec

OP = "attention.dsv41_mla_prolog"
HIDDEN = 5120
Q_LORA = 1280
KV_LORA = 512
HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
Q_OUT = HEADS * HEAD_DIM
CACHE_PAGES = 22594
CACHE_ROWS = 128
W_QB_NZ_SHAPE = (Q_OUT // 32, Q_LORA // 16, 16, 32)
MIN_T = 1
MAX_T = 4096

INPUT_NAMES = (
    "x", "w_qa", "gamma_qa", "w_qb", "w_qb_scale", "w_kv", "gamma_kv",
    "rope_cos", "rope_sin", "slot_indices", "kv_cache",
)
OUTPUT_NAMES = ("q_output",)
FUSION_OPS = (
    "dual_projection", "rms_norm", "dynamic_quantize",
    "quant_matmul", "partial_rope", "kv_cache_scatter",
)
# w_qb is an INT8 weight with one static BF16 scale per logical output channel.
# The other quantized operand is produced inside the fused chain, so it has no
# separate public Tensor binding.
QUANTIZATION = (
    (3, QuantSpec(
        scheme="symmetric", storage_dtype="int8", compute_dtype="int8",
        accumulator_dtype="int32", scale_dtype="bfloat16",
        granularity="channel", axis=(1,), physical_layout="NZ",
    )),
)


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def infer(arguments, layouts):
    """Validate metadata only; never read Tensor values or allocate outputs."""
    x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin, slots, cache = (
        arguments[name] for name in INPUT_NAMES
    )
    t = int(x.shape[0]) if len(x.shape) == 2 else -1
    if t < MIN_T or t > MAX_T:
        raise ValueError("x must have shape [T, 5120] with T in [1, 4096]")
    expected = {
        "x": (t, HIDDEN),
        "w_qa": (Q_LORA, HIDDEN),
        "gamma_qa": (Q_LORA,),
        "w_qb": W_QB_NZ_SHAPE,
        "w_qb_scale": (Q_OUT,),
        "w_kv": (KV_LORA, HIDDEN),
        "gamma_kv": (KV_LORA,),
        "rope_cos": (t, 1, 1, ROPE_DIM),
        "rope_sin": (t, 1, 1, ROPE_DIM),
        "slot_indices": (t, 2),
        "kv_cache": (CACHE_PAGES, CACHE_ROWS, KV_LORA),
    }
    tensors = {
        "x": x, "w_qa": w_qa, "gamma_qa": gamma_qa, "w_qb": w_qb,
        "w_qb_scale": w_qb_scale, "w_kv": w_kv, "gamma_kv": gamma_kv,
        "rope_cos": rope_cos, "rope_sin": rope_sin, "slot_indices": slots,
        "kv_cache": cache,
    }
    for name, shape in expected.items():
        if _shape(tensors[name]) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {_shape(tensors[name])}")
    dtypes = {
        "x": "bfloat16", "w_qa": "bfloat16", "gamma_qa": "bfloat16",
        "w_qb": "int8", "w_qb_scale": "bfloat16", "w_kv": "bfloat16",
        "gamma_kv": "bfloat16", "rope_cos": "float32", "rope_sin": "float32",
        "slot_indices": "int32", "kv_cache": "bfloat16",
    }
    for name, dtype in dtypes.items():
        if _dtype(tensors[name]) != dtype:
            raise ValueError(f"{name} requires {dtype}, got {_dtype(tensors[name])}")
    if layouts is not None:
        # FRACTAL_NZ is the only non-ND public input; it is the packed INT8
        # weight. kv_cache is logically NCL but physically contiguous ND.
        expected_layouts = {name: "ND" for name in INPUT_NAMES}
        expected_layouts["w_qb"] = "NZ"
        for name, layout in expected_layouts.items():
            if name in layouts and layouts[name] != layout:
                raise ValueError(f"{name} requires {layout} physical layout")
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (OutputSpec((t, HEADS, HEAD_DIM), "bfloat16", x.device),),
        {},
        InferenceSpec(
            compute_dtype="bfloat16", accumulator_dtype="float32",
            quantization=QUANTIZATION, fusion_ops=FUSION_OPS,
            features=(
                ("normalization.kind", "rms"),
                ("normalization.affine", True),
            ),
        ),
    )


def validate_values(arguments):
    """Reject cache addresses and encoder values the native kernel cannot use.

    Large projection operands are not copied to the host here; the independent
    CPU cases and the standalone validation cover their finiteness. This hook
    checks only the small public Tensor values.
    """
    import torch

    x = arguments["x"]
    t = int(x.shape[0])
    if t < MIN_T or t > MAX_T:
        raise ValueError("T must be in [1, 4096]")
    slots = arguments["slot_indices"].detach().cpu()
    if bool((slots[:, 0] < 0).any()) or bool((slots[:, 0] >= CACHE_PAGES).any()):
        raise ValueError("slot page index is out of range")
    if bool((slots[:, 1] < 0).any()) or bool((slots[:, 1] >= CACHE_ROWS).any()):
        raise ValueError("slot row index is out of range")
    if torch.unique(slots, dim=0).shape[0] != t:
        raise ValueError("slot_indices must be unique within one call")
    cos = arguments["rope_cos"].detach().cpu().float().reshape(t, ROPE_DIM)
    sin = arguments["rope_sin"].detach().cpu().float().reshape(t, ROPE_DIM)
    if not torch.equal(cos[:, 0::2], cos[:, 1::2]):
        raise ValueError("rope_cos adjacent even/odd values must be equal")
    if not torch.equal(sin[:, 0::2], sin[:, 1::2]):
        raise ValueError("rope_sin adjacent even/odd values must be equal")
    scale = arguments["w_qb_scale"].detach().cpu().float()
    if bool((scale < 0).any()):
        raise ValueError("w_qb_scale must be non-negative")
    for name in ("gamma_qa", "gamma_kv", "rope_cos", "rope_sin", "w_qb_scale"):
        value = arguments[name].detach().cpu().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")


OPERATOR = OperatorDefinition(
    name="dsv41_mla_prolog",
    domain="attention",
    op=OP,
    schema=(
        "(Tensor x, Tensor w_qa, Tensor gamma_qa, Tensor w_qb, "
        "Tensor w_qb_scale, Tensor w_kv, Tensor gamma_kv, Tensor rope_cos, "
        "Tensor rope_sin, Tensor slot_indices, Tensor kv_cache) -> Tensor"
    ),
    output_names=OUTPUT_NAMES,
    infer=infer,
    validate_values=validate_values,
    description=(
        "DS4.1 MLA prolog fusion with fixed H=5120, Q_LORA=1280, KV_LORA=512, "
        "N=64, D_NOPE=448 and D_ROPE=64. Writes q_output[T,64,512] and updates "
        "the addressed kv_cache rows in place; unaddressed rows are sentinels. "
        "BF16 materialization points and INT8 dynamic quantization are frozen. "
        "T is in [1,4096] and each call is one launch. Graph capture is not "
        "declared. The in-place cache mutation is not yet expressible in the "
        "generated schema path; see the submission gap analysis."
    ),
)
