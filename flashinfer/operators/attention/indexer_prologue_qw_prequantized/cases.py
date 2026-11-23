"""Independent CPU oracle and samples for the caller-prequantized QW contract.

The NPU weight is only a physical NZ representation of an ordinary INT8 ND
logical weight. The oracle reads that logical INT8 weight (and the caller's
scale) on CPU and never asks torch_npu to compute a reference.
"""

import math

from ..indexer_prologue_qw.cases import (
    _dynamic_quant,
    _quantize_query,
    _rope_tables,
    _rope_tail,
    _unit_amax,
    _w_projection,
)
from ..indexer_prologue_qw.operator import (
    DEFAULT_SOFTMAX_SCALE,
    DIM_Q,
    DIM_QR,
    DIM_X,
    HEAD_DIM,
    N_HEADS,
    ROPE_DIM,
)


def golden(torch, x, qr, weight_q, weight_scale, ww, rope_sin, rope_cos,
           softmax_scale=DEFAULT_SOFTMAX_SCALE):
    """CPU reference over the same logical INT8 ND weight the launch consumes."""
    tokens = qr.shape[0]
    activation_q, activation_scale = _dynamic_quant(torch, qr)
    accumulated = torch.matmul(activation_q.int(), weight_q.int())
    dequant = (accumulated.float() * (activation_scale[:, None] * weight_scale[None, :]))
    projected = dequant.to(torch.bfloat16).view(tokens, N_HEADS, HEAD_DIM)
    roped = _rope_tail(torch, projected, rope_sin, rope_cos)
    q8, q_scale = _quantize_query(torch, roped)
    w = _w_projection(torch, x, ww, softmax_scale)
    return q8, q_scale, w


def sample(torch, torch_npu, *, tokens=48, seed=20261119,
           softmax_scale=DEFAULT_SOFTMAX_SCALE):
    from flashinfer.attention.indexer_prologue import prepare_indexer_prologue_qw_weight

    generator = torch.Generator(device="cpu").manual_seed(seed + tokens)
    x = torch.randn(tokens, DIM_X, generator=generator).to(torch.bfloat16)
    qr = _unit_amax(torch, torch.randn(tokens, DIM_QR, generator=generator).to(torch.bfloat16))
    wqb = _unit_amax(torch, (torch.randn(DIM_Q, DIM_QR, generator=generator)
                             / math.sqrt(DIM_QR)).to(torch.bfloat16))
    ww = (torch.randn(N_HEADS, DIM_X, generator=generator) / math.sqrt(DIM_X)).to(torch.bfloat16)
    rope_sin, rope_cos = _rope_tables(torch, tokens, generator)

    wqb_nz, wqb_scale = prepare_indexer_prologue_qw_weight(wqb.npu())
    # Independent CPU weight oracle: do not derive the expected values from
    # the helper under test. The normalized rows make the quantizer exact.
    weight_q, weight_scale = _dynamic_quant(torch, wqb)
    logical_weight = weight_q.t().contiguous()
    expected = golden(torch, x, qr, logical_weight, weight_scale, ww,
                      rope_sin, rope_cos, softmax_scale)
    args = (x.npu(), qr.npu(), wqb_nz, wqb_scale, ww.npu(), rope_sin.npu(), rope_cos.npu())
    return args, dict(softmax_scale=softmax_scale), expected


CASE_PARAMETERS = tuple(
    dict(tokens=tokens, seed=20261119)
    for tokens in (1, 3, 48, 49, 72, 96, 128, 129)
)
TOLERANCES = ((1.0, 0.0), (9e-2, 2 ** -10), (9e-2, 2 ** -10))
