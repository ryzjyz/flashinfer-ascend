"""Public caller-prequantized contract for attention.indexer_prologue_qw_prequantized."""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.indexer_prologue_qw_prequantized"
N_HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
DIM_X = 5120
DIM_QR = 1280
DIM_Q = N_HEADS * HEAD_DIM
DEFAULT_SOFTMAX_SCALE = 128 ** -0.5


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def infer(arguments, layouts):
    x, qr, wqb_nz, wqb_scale, ww, rope_sin, rope_cos = (
        arguments[name]
        for name in ("x", "qr", "wqb_nz", "wqb_scale", "ww", "rope_sin", "rope_cos")
    )
    softmax_scale = arguments["softmax_scale"]
    tokens = int(x.shape[0]) if len(x.shape) else 0
    if tokens < 1:
        raise ValueError("IndexerPrologueQWPrequantized requires T >= 1")
    expected = (
        ("x", x, "bfloat16", (tokens, DIM_X)),
        ("qr", qr, "bfloat16", (tokens, DIM_QR)),
        ("wqb_nz", wqb_nz, "int8", (DIM_QR, DIM_Q)),
        ("wqb_scale", wqb_scale, "float32", (DIM_Q,)),
        ("ww", ww, "bfloat16", (N_HEADS, DIM_X)),
        ("rope_sin", rope_sin, "float32", (tokens, ROPE_DIM)),
        ("rope_cos", rope_cos, "float32", (tokens, ROPE_DIM)),
    )
    for name, tensor, dtype, shape in expected:
        if _dtype(tensor) != dtype:
            raise ValueError("%s must be %s" % (name, dtype))
        if tuple(tensor.shape) != shape:
            raise ValueError("%s must have shape %s; got %s" % (name, shape, tuple(tensor.shape)))
    if layouts is not None:
        if layouts.get("wqb_nz") != "NZ":
            raise ValueError("wqb_nz must carry physical FRACTAL_NZ")
        if any(layouts.get(name) != "ND" for name in
               ("x", "qr", "wqb_scale", "ww", "rope_sin", "rope_cos")):
            raise ValueError("prequantized QW activations require physical ND")
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise ValueError("softmax_scale must be a finite float")
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be a finite float")
    inference = InferenceSpec(
        compute_dtype="int8", accumulator_dtype="int32",
        fusion_ops=("quantized_matmul", "rope", "dynamic_quantize"),
        auxiliary_outputs=("q_scale", "w"),
    )
    return Invocation(
        (x, qr, wqb_nz, wqb_scale, ww, rope_sin, rope_cos),
        (
            OutputSpec((tokens, N_HEADS, HEAD_DIM), "int8", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
        ),
        {"softmax_scale": float(softmax_scale)},
        inference,
        specializations=(("quant.weight", ("caller_prequantized_nz",)),),
    )


def validate_values(arguments):
    if not math.isfinite(float(arguments["softmax_scale"])):
        raise ValueError("softmax_scale must be a finite float")


OPERATOR = OperatorDefinition(
    name="indexer_prologue_qw_prequantized", domain="attention", op=OP,
    schema="(Tensor x, Tensor qr, Tensor wqb_nz, Tensor wqb_scale, Tensor ww, "
           "Tensor rope_sin, Tensor rope_cos, "
           "float softmax_scale=0.08838834764831845) -> (Tensor, Tensor, Tensor)",
    output_names=("q", "q_scale", "w"), infer=infer, validate_values=validate_values,
    description="Mixed Q projection over caller-prepared INT8 NZ weights, query "
                "RoPE/quantization and W projection. No forward weight preprocessing.",
)
