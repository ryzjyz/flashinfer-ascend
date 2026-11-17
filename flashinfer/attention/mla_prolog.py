"""Public facade for the A3 DS4.1 MLA prolog fusion.

The fused chain writes ``q_output[T,64,512]`` and updates the addressed rows of
``kv_cache`` in place. ``kv_cache`` is declared as a mutated argument on the
Torch custom-op bridge; unaddressed rows are sentinels and must not change.
This eager entry is not graph capturable.
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
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Run the fixed DS4.1 MLA prolog chain on Ascend910C.

    ``x`` is BF16 ``[T,5120]`` with T in [1,4096]; ``w_qb`` is INT8 FRACTAL_NZ
    ``[1024,80,16,32]``; ``kv_cache`` is BF16 ``[22594,128,512]`` and is updated
    in place for the rows named by ``slot_indices[T,2]``. Returns
    ``q_output[T,64,512]`` BF16. The dispatcher rejects unsupported dtypes,
    shapes, physical layouts, SoC and missing AOT artifacts.
    """
    import torch
    from ..aclnn._ops import ensure_registered

    ensure_registered()
    options = dict(
        backend=backend.value if isinstance(backend, Backend) else backend,
        kernel_id=kernel_id,
    )
    if out is None:
        return torch.ops.flashinfer.mla_prolog(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, **options,
        )
    torch.ops.flashinfer.mla_prolog_out(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, out, **options,
    )
    return out
