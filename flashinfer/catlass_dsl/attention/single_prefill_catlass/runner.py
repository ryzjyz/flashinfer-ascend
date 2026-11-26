"""Compile with CATLASS DSL and launch on the calling TorchNPU stream."""

from threading import Lock

from flashinfer import config
from flashinfer.catlass_dsl._launcher import PreparedLaunch


class CatlassDslPrefillRunner:
    def __init__(self, _descriptor):
        self._compiled = {}
        self._lock = Lock()

    def __call__(self, q, k, v, out):
        import torch

        if config.jit_disabled():
            raise RuntimeError("CATLASS DSL requires JIT (FLASHINFER_NPU_JIT_DISABLE=1)")
        try:
            from catlass import tla
        except ModuleNotFoundError as error:
            if error.name != "catlass":
                raise
            raise RuntimeError(
                "CATLASS DSL requires ascend-catlass-dsl. Install the 'catlass-dsl' extra."
            ) from error
        from catlass.tla.runtime import from_dlpack

        from .kernels import Q_TILE, dense_prefill

        # The common prefill entry has already checked the NHD storage and overlap.
        tensors = (q, k, v, out)
        q_length, heads, _ = q.shape
        shape = (q_length, k.shape[0], heads)
        key = (q.device.index, q.dtype)
        with torch.npu.device(q.device):
            buffers = tuple(
                from_dlpack(tensor, layout_tag=tla.arch.RowMajor, origin_shape=(tensor.numel(),))
                .mark_layout_dynamic(leading_dim=0)
                for tensor in tensors
            )
            arguments = (*buffers, *shape)
            prepared = self._compiled.get(key)
            if prepared is None:
                with self._lock:
                    prepared = self._compiled.get(key)
                    if prepared is None:
                        compiled = tla.compile(dense_prefill, *arguments, q.dtype == torch.float16,
                                               options="--npu-arch 3510")
                        cores = torch.npu.get_device_properties(q.device).cube_core_num
                        prepared = self._compiled[key] = (PreparedLaunch(compiled), cores)
            launch, cores = prepared
            launch(arguments, tensors, block_num=min(cores, (q_length // Q_TILE) * heads))
