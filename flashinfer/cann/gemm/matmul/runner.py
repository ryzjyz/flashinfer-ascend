"""Execution adapter for gemm.matmul (cann)."""

from __future__ import annotations

def _matmul_cann(a, b, out):
    """Write directly into the common output through the prebuilt CANN op."""
    import torch
    from flashinfer._matmul import validate_tensors

    validate_tensors(a, b, out)
    with torch.npu.device(a.device):
        torch.mm(a, b, out=out)

run = _matmul_cann
