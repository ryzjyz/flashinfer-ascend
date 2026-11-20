"""Public facade for the A3 DS4.1 MLA prolog fusion.

The fused chain returns ``q_output[T,q_head_num,512]`` and the required
``qr[T,1280] = q_norm(wq_a(x))``, and updates the addressed rows of
``kv_cache`` in place. ``q_head_num`` is the local Q head count in ``[1,64]``
(default ``64``); TP sharding is the caller's responsibility and the operator
never communicates across cards. ``qr`` is not sharded. ``rms_norm_eps``
parameterizes both RMSNorm branches (default ``1e-6``; DS4.1 callers may pass
``1e-20``). ``kv_cache`` is declared as a mutated argument on the Torch
custom-op bridge; unaddressed rows are sentinels and must not change. This
eager entry is not graph capturable.
"""

from __future__ import annotations

from typing import Optional, Union

from ..runtime import Backend


def mla_prolog(
    x,
    w_qa,
    gamma_qa,
    w_qb,
    w_qb_scale,
    w_kv,
    gamma_kv,
    rope_cos,
    rope_sin,
    slot_indices,
    kv_cache,
    out=None,
    qr_out=None,
    *,
    rms_norm_eps: float = 1e-6,
    q_head_num: int = 64,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Run the DS4.1 MLA prolog chain on Ascend910C.

    ``x`` is BF16 ``[T,5120]`` with T in [1,4096]; ``q_head_num`` is the local
    Q head count in [1,64] (default 64, for TP sharding); ``w_qb`` is INT8
    FRACTAL_NZ ``[q_head_num*16,80,16,32]`` and ``w_qb_scale`` is
    ``[q_head_num*512]``; ``kv_cache`` is BF16 ``[22594,128,512]`` and is
    updated in place for the rows named by ``slot_indices[T,2]``. Returns
    ``(q_output[T,q_head_num,512], qr[T,1280])``, both BF16; ``qr`` is not
    sharded by ``q_head_num``. ``rms_norm_eps`` must be finite and strictly
    positive. Passing ``out`` and/or ``qr_out`` writes the supplied tensors
    (the other output is allocated) and returns the pair. The dispatcher
    rejects unsupported dtypes, shapes, physical layouts, SoC and missing AOT
    artifacts.
    """
    import math
    import torch
    from ..aclnn._ops import ensure_registered

    # Reject non-numeric or non-finite eps before the Torch bridge coerces it.
    if (type(rms_norm_eps) not in (int, float)
            or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0):
        raise ValueError("rms_norm_eps must be finite and > 0")
    # Reject bool/float/out-of-range head counts before the Torch bridge
    # coerces the int attribute.
    if type(q_head_num) is not int or not 1 <= q_head_num <= 64:
        raise ValueError("q_head_num must be an integer in [1, 64]")
    ensure_registered()
    options = dict(
        backend=backend.value if isinstance(backend, Backend) else backend,
        kernel_id=kernel_id,
        rms_norm_eps=rms_norm_eps,
        q_head_num=q_head_num,
    )
    if out is None and qr_out is None:
        return torch.ops.flashinfer.mla_prolog(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, **options,
        )
    if isinstance(out, (tuple, list)):
        if qr_out is not None or len(out) != 2:
            raise ValueError("out must be (q_output, qr) or pass qr_out separately")
        out, qr_out = out
    t = int(x.shape[0])
    if out is None:
        out = torch.empty(
            (t, q_head_num, 512), dtype=torch.bfloat16, device=x.device)
    if qr_out is None:
        qr_out = torch.empty(
            (t, 1280), dtype=torch.bfloat16, device=x.device)
    torch.ops.flashinfer.mla_prolog_out(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, out, qr_out, **options,
    )
    return out, qr_out
