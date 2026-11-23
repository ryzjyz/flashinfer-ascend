"""Public dense-BF16 weight contract for attention.indexer_prologue_qw."""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.indexer_prologue_qw"
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
    x, qr, wqb, ww, rope_sin, rope_cos = (
        arguments[name] for name in ("x", "qr", "wqb", "ww", "rope_sin", "rope_cos")
    )
    softmax_scale = arguments["softmax_scale"]
    tokens = int(x.shape[0]) if len(x.shape) else 0
    expected = (
        ("x", x, "bfloat16", (tokens, DIM_X)),
        ("qr", qr, "bfloat16", (tokens, DIM_QR)),
        ("wqb", wqb, "bfloat16", (DIM_Q, DIM_QR)),
        ("ww", ww, "bfloat16", (N_HEADS, DIM_X)),
        ("rope_sin", rope_sin, "float32", (tokens, ROPE_DIM)),
        ("rope_cos", rope_cos, "float32", (tokens, ROPE_DIM)),
    )
    if tokens < 1:
        raise ValueError("IndexerPrologueQW requires T >= 1")
    for name, tensor, dtype, shape in expected:
        if _dtype(tensor) != dtype:
            raise ValueError("%s must be %s" % (name, dtype))
        if tuple(tensor.shape) != shape:
            raise ValueError("%s must have shape %s; got %s" % (name, shape, tuple(tensor.shape)))
    if layouts is not None and any(layouts.get(name) != "ND" for name in
                                   ("x", "qr", "wqb", "ww", "rope_sin", "rope_cos")):
        raise ValueError("IndexerPrologueQW dense weights and activations require physical ND")
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise ValueError("softmax_scale must be a finite float")
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be a finite float")
    inference = InferenceSpec(
        compute_dtype="int8", accumulator_dtype="int32",
        fusion_ops=("dynamic_quantize", "quantized_matmul", "rope", "dynamic_quantize"),
        auxiliary_outputs=("q_scale", "w"),
    )
    return Invocation(
        (x, qr, wqb, ww, rope_sin, rope_cos),
        (
            OutputSpec((tokens, N_HEADS, HEAD_DIM), "int8", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
        ),
        {"softmax_scale": float(softmax_scale)},
        inference,
        specializations=(("quant.activation", ("per_token_int8",)),
                          ("quant.weight", ("per_channel_int8",))),
    )


def validate_values(arguments):
    softmax_scale = arguments["softmax_scale"]
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be a finite float")


OPERATOR = OperatorDefinition(
    name="indexer_prologue_qw", domain="attention", op=OP,
    schema="(Tensor x, Tensor qr, Tensor wqb, Tensor ww, Tensor rope_sin, Tensor rope_cos, "
           "float softmax_scale=0.08838834764831845) -> (Tensor, Tensor, Tensor)",
    output_names=("q", "q_scale", "w"), infer=infer, validate_values=validate_values,
    description="Mixed W8A8 Q projection, query RoPE/quantization and W projection. "
                "The dense BF16 weight is dynamically quantized inside the launch.",
)
