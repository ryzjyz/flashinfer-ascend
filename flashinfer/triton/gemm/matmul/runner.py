from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import RLock
from flashinfer._matmul import TRITON_KERNEL_ID, validate_tensors
from flashinfer.jit._triton_abi import record_tensor_streams, task_queue_enabled
from flashinfer.runtime import (
    Backend,
    CompileMode,
    KernelDescriptor,
    SchemaError,
    SourceRef,
)
from flashinfer.tuning.identity import execution_sources, fingerprint, runtime_identity

_CONSTANTS = dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
_OPTIONS = dict(num_warps=4, num_stages=1)


class TritonMatmulRunner:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self._modules = {}
        self._lock = RLock()
        self.build_requests = 0
        self.launches = 0

    def spec(self):
        return self._spec(task_queue_enabled())

    @lru_cache(maxsize=2)
    def _spec(self, queued):
        from flashinfer.jit import JitEnvironment, TritonBackend

        environment = runtime_identity()
        signature = dict(a="*bf16", b="*bf16", out="*bf16", M="i32", N="i32", K="i32")
        build_environment = JitEnvironment(
            "Ascend950",
            "Ascend950",
            fingerprint(environment["cann"]),
            "triton-ascend",
            environment["triton"],
            environment["torch"],
            environment["torch_npu"],
            environment["python_abi"],
        )
        return TritonBackend().declare(
            self.descriptor,
            task_queue=queued,
            target_soc="Ascend950",
            output_path="matmul/kernel.bin",
            entry_point="_matmul_kernel",
            environment_fingerprint=build_environment.fingerprint,
            specialization_fingerprint=fingerprint(
                dict(constants=_CONSTANTS, options=_OPTIONS)
            ),
            compile_options=(
                "signature=" + json.dumps(signature, sort_keys=True),
                "constexprs=" + json.dumps(_CONSTANTS, sort_keys=True),
                "options=" + json.dumps(_OPTIONS, sort_keys=True),
                'binary={"mode":"mix"}',
            ),
        )

    def tuning_identity(self, signature, hardware):
        return self._execution_identity(task_queue_enabled())

    @lru_cache(maxsize=2)
    def _execution_identity(self, queued):
        return fingerprint(
            dict(
                spec=self._spec(queued).fingerprint,
                runtime=runtime_identity(),
                runner=execution_sources(
                    (
                        "triton/gemm/matmul/runner.py",
                        "_matmul.py",
                        "jit/_triton_compile.py",
                        "jit/_triton_abi.py",
                        "jit/loading.py",
                    )
                ),
            )
        )

    def prepare(self, device_index):
        queued = task_queue_enabled()
        key = (device_index, queued)
        module = self._modules.get(key)
        if module is not None:
            return module
        from flashinfer.jit import TritonBackend, build_and_load

        with self._lock:
            if key not in self._modules:
                source = (
                    Path(__file__).resolve().parents[3] / self.descriptor.source.locator
                )
                if (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    != self.descriptor.source.digest
                ):
                    raise SchemaError(
                        "Triton Matmul SourceRef digest mismatch; refresh the declaration"
                    )
                self._modules[key] = build_and_load(
                    self._spec(queued), backend=TritonBackend()
                )
                self.build_requests += 1
            return self._modules[key]

    def __call__(self, a, b, out):
        import torch

        validate_tensors(a, b, out)
        device = a.device.index
        if device != torch.npu.current_device():
            with torch.npu.device(device):
                return self._launch(a, b, out, device)
        return self._launch(a, b, out, device)

    def _launch(self, a, b, out, device):
        module = self.prepare(device)
        record_tensor_streams((a, b, out))
        m, k = a.shape
        n = b.shape[1]
        block_m, block_n = _CONSTANTS["BLOCK_M"], _CONSTANTS["BLOCK_N"]
        module.opaque_module[
            ((m + block_m - 1) // block_m, (n + block_n - 1) // block_n)
        ](a, b, out, m, n, k)
        self.launches += 1
