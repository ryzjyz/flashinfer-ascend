"""Framework-neutral paged KV append API matching pinned FlashInfer."""

from __future__ import annotations

from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .runtime.runner import run_operator


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
) -> None:
    return run_operator(
        "page.append_paged_kv_cache", (append_key,),
        (
            append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices,
            kv_indptr, kv_last_page_len, kv_layout,
        ),
        layouts=(kv_layout,),
    )


# Definitions and implementation choices are collected by the framework.
from flashinfer.operators.catalog import install_exports as _install_operator_exports
_install_operator_exports(globals(), domain='page')
