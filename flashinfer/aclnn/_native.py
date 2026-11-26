"""Compiler-free loading and calls into the verified Torch/ACLNN adapter."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import NamedTuple

_lock = RLock()
_owners = {}


class _Binding(NamedTuple):
    module: object
    library: object
    run: object


def adapter_directory():
    from flashinfer._resources import data_root

    return Path(os.environ.get("FLASHINFER_ACLNN_ADAPTER_PATH", data_root() / "aclnn"))


def verified_adapter(directory=None):
    """Verify prebuilt bytes and build identity without loading the library."""
    import torch
    import torch_npu

    directory = adapter_directory() if directory is None else Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(
            "ACLNN Torch adapter is not bundled; build a normal AOT wheel or run "
            "python -m flashinfer.aclnn.build --output-dir flashinfer/data/aclnn"
        )
    if manifest.is_symlink():
        raise RuntimeError("ACLNN Torch adapter manifest must not be a symlink")
    record = json.loads(manifest.read_text())
    if (
        record.get("schema_version") != 1
        or record.get("library") != "libflashinfer_aclnn.so"
        or record.get("torch_version") != torch.__version__
        or record.get("torch_npu_version") != torch_npu.__version__
    ):
        raise RuntimeError(
            "ACLNN Torch adapter Torch/torch_npu build identity mismatch"
        )
    library = directory / record["library"]
    if library.is_symlink():
        raise RuntimeError("ACLNN Torch adapter library must not be a symlink")
    if hashlib.sha256(library.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("ACLNN Torch adapter library digest mismatch")
    if record.get("submission") != "torch_npu_task_queue":
        raise RuntimeError("ACLNN Torch adapter lacks queued submission; rebuild the adapter or the AOT wheel")
    from .._resources import data_root, source_root
    header = (source_root() or data_root() / "build") / "csrc/aclnn_torch_adapter/call.h"
    if record.get("call_header_sha256") != hashlib.sha256(header.read_bytes()).hexdigest():
        raise RuntimeError("ACLNN Torch adapter call ABI mismatch; rebuild the adapter")
    return library, manifest


@lru_cache(maxsize=1)
def load_adapter():
    import ctypes as C
    import torch

    library, _ = verified_adapter()
    torch.ops.load_library(str(library))
    api = C.CDLL(str(library))
    api.fi_bind_package.argtypes = [C.c_char_p, C.c_void_p, C.c_void_p, C.c_void_p]
    api.fi_bind_package.restype = C.c_int
    api.fi_aclnn_last_error.argtypes = []
    api.fi_aclnn_last_error.restype = C.c_char_p
    api.fi_aclnn_last_workspace_size.argtypes = [C.c_char_p, C.POINTER(C.c_uint64)]
    api.fi_aclnn_last_workspace_size.restype = C.c_int
    if hasattr(getattr(torch.utils, "_python_dispatch", None), "is_in_torch_dispatch_mode"):
        from ..runtime import signature
        signature._batch_format_query = (torch, torch.ops.flashinfer_aclnn._storage_formats.default)
    return api


def prepare(module, recipe):
    """Bind one verified module once; invocation only supplies Tensor lists."""
    import ctypes as C
    import torch

    token = recipe.fingerprint + ":" + str(id(module.api))
    with _lock:
        if token in _owners:
            return _owners[token].run
        api = load_adapter()
        # The store/release verifies the companion bytes together with the vendor package.
        companion = C.CDLL(str(module.call_path), mode=C.RTLD_LOCAL)
        identity = companion.fi_submit_identity
        identity.argtypes, identity.restype = [], C.c_char_p
        if identity() != recipe.fingerprint.encode("ascii"):
            raise RuntimeError("ACLNN submission library does not match its verified recipe")
        status = api.fi_bind_package(token.encode(), companion.fi_submit,
                                     getattr(module.api, recipe.host_entry + "GetWorkspaceSize"),
                                     getattr(module.api, recipe.host_entry))
        if status:
            raise RuntimeError(api.fi_aclnn_last_error().decode())
        inputs = sum(p.role == "input" for p in recipe.tensor_parameters)
        grouped = any(p.tensor_list or p.optional for p in recipe.tensor_parameters)
        dtype_positions = tuple(i for i, attr in enumerate(recipe.attributes) if attr.dtype == "dtype")
        uint64_positions = tuple(i for i, attr in enumerate(recipe.attributes) if attr.dtype == "uint64")
        operation = torch.ops.flashinfer_aclnn._execute.default

        def run(tensors, attributes=()):
            if dtype_positions or uint64_positions:
                attributes = list(attributes)
                for i in dtype_positions:
                    attributes[i] = getattr(torch, attributes[i])
                for i in uint64_positions:
                    if attributes[i] >= 1 << 63:
                        attributes[i] -= 1 << 64  # IValue stores integers in a signed 64-bit slot.
            if not grouped:
                return operation(token, tensors[:inputs], tensors[inputs:], attributes, ())
            flat_inputs, flat_outputs, widths = [], [], []
            for p, value in zip(recipe.tensor_parameters, tensors):
                dest = flat_inputs if p.role == "input" else flat_outputs
                if value is None:
                    widths.append(-1)
                elif p.tensor_list:
                    widths.append(len(value))
                    dest.extend(value)
                else:
                    widths.append(1)
                    dest.append(value)
            return operation(token, flat_inputs, flat_outputs, attributes, widths)

        _owners[token] = _Binding(module, companion, run)
        return run


def last_workspace_size(binding, recipe):
    """Observe the last real GetWorkspaceSize result of a prepared binding."""
    import ctypes as C

    api = load_adapter()
    size = C.c_uint64()
    token = recipe.fingerprint + ":" + str(id(binding.module.api))
    if api.fi_aclnn_last_workspace_size(token.encode(), C.byref(size)):
        raise RuntimeError(api.fi_aclnn_last_error().decode())
    return size.value
