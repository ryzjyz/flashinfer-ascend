"""Host launcher and device symbol contracts for Triton artifacts."""

import os

from .core import _canonical_hash
from ._elf import elf_symbols
from ..runtime.schema import SchemaError


_LAUNCHER_DIGESTS = {mode: _canonical_hash({"task_queue": mode}) for mode in (False, True)}


def task_queue_enabled():
    """Match the Triton Ascend launcher's build-time option."""
    return os.environ.get("TRITON_ENABLE_TASKQUEUE", "true").lower() in ("true", "1")


def launcher_input(queued):
    return ("triton_launcher", _LAUNCHER_DIGESTS[queued])


def spec_task_queue(spec):
    digest = dict(spec.input_fingerprints).get("triton_launcher")
    for mode, expected in _LAUNCHER_DIGESTS.items():
        if digest == expected:
            return mode
    raise SchemaError("Triton spec has no recognized launcher policy; rebuild the artifact")


def direct_launch_stream(stream):
    """TorchNPU drains pending Host submissions when exporting this raw handle."""
    return stream.npu_stream


def record_tensor_streams(tensors, *, drain_host_queue=False):
    """Record storage on the calling thread's current stream for this invocation."""
    import torch

    device = tensors[0].device
    accelerator = getattr(torch, "accelerator", None)
    get_stream = getattr(accelerator, "current_stream", None)
    get_accelerator = getattr(accelerator, "current_accelerator", None)
    active = get_accelerator() if get_stream and get_accelerator and not drain_host_queue else None
    if active is not None and active.type == device.type:
        stream = get_stream(device.index)
    else:
        stream = torch.npu.current_stream(device)
    for tensor in tensors:
        tensor.record_stream(stream)
    if drain_host_queue:
        direct_launch_stream(stream)


def triton_binary_mode(config):
    declaration = config.get("binary", {"mode": "single"})
    if set(declaration) != {"mode"} or declaration["mode"] not in {"single", "mix"}:
        raise ValueError("Triton binary must declare exactly mode=single or mode=mix")
    return declaration["mode"]


def verify_triton_symbols(payload, entry, mode="single"):
    """Mixed ABI requires BOTH derived device entries, never an arbitrary suffix."""
    if mode not in {"single", "mix"}:
        raise SchemaError("unknown Triton binary mode")
    required = ({entry} if mode == "single" else
                {entry + "_mix_aic", entry + "_mix_aiv"})
    symbols = elf_symbols(payload, machine=0x1029, elf_type=2)
    if not required <= symbols:
        raise SchemaError("Triton %s ELF is missing defined entry symbols %r" % (
            mode, sorted(required - symbols)))
    return symbols
