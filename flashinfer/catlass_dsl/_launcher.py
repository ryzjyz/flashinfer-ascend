"""Submit CATLASS DSL kernels through TorchNPU's native task queue."""

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
from threading import Lock


_MODULE = "_flashinfer_catlass_dsl_launcher"
_SOURCE = "csrc/catlass_dsl/launcher.cpp"
_launcher_lock = Lock()


class _LauncherBuild:
    backend_id = "catlass_dsl_launcher"

    def __init__(self):
        from flashinfer._resources import data_root, source_root

        root = source_root() or (data_root() / "build")
        self.source = (root / _SOURCE).read_bytes()
        cann_root = os.environ.get("ASCEND_HOME_PATH")
        if not cann_root:
            raise RuntimeError("CATLASS DSL launcher requires ASCEND_HOME_PATH from CANN set_env.sh")
        self.cann = Path(cann_root).resolve(strict=True)

    def declare(self):
        from flashinfer.aclnn.build import torch_abi
        from flashinfer.jit import JitSpec
        from flashinfer.runtime import SourceRef
        from flashinfer.tuning.identity import fingerprint, runtime_identity

        source = SourceRef("c++", _SOURCE, hashlib.sha256(self.source).hexdigest())
        return JitSpec(
            name=self.backend_id,
            domain="host",
            generator_id=self.backend_id,
            generator_version="1",
            target_soc="host",
            environment_fingerprint=fingerprint({
                "platform": sysconfig.get_platform(),
                "python_abi": sysconfig.get_config_var("SOABI"),
                "torch_abi": torch_abi(),
                "cann_root": str(self.cann),
                "cann": runtime_identity()["cann"],
            }),
            specialization_fingerprint=source.digest,
            input_fingerprints=(("build_driver", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),),
            kernel_id="catlass_dsl/host_launcher",
            source=source,
            output_path=_MODULE + ".so",
            entry_points=("PyInit_" + _MODULE,),
        )

    def compile(self, spec, *, build_directory):
        import torch
        import torch_npu
        from torch.utils.cpp_extension import include_paths, library_paths

        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("Building the CATLASS DSL launcher requires a C++17 compiler")
        source = build_directory / "launcher.cpp"
        source.write_bytes(self.source)
        npu = Path(torch_npu.__file__).resolve().parent
        includes = [*include_paths(), npu / "include", self.cann / "include", sysconfig.get_path("include")]
        libraries = [*library_paths(), npu / "lib", self.cann / "lib64"]
        command = [
            compiler, "-std=c++17", "-O2", "-g0", "-fPIC", "-fvisibility=hidden", "-shared",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
            str(source), "-o", str(build_directory / spec.output_path),
            *(f"-I{path}" for path in includes), *(f"-L{path}" for path in libraries),
            "-ltorch_python", "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu", "-lascendcl",
        ]
        with (build_directory / "build.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("CATLASS DSL launcher compilation failed")


@lru_cache(maxsize=1)
def _load_launcher():
    from flashinfer.jit.build import _build_record
    from flashinfer.jit.cache import DiskJitBuildStore

    builder = _LauncherBuild()
    store = DiskJitBuildStore()
    record = _build_record(builder.declare(), backend=builder, store=store)
    spec = importlib.util.spec_from_file_location(_MODULE, store.artifact_path(record))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreparedLaunch:
    """Bind a compiled DSL kernel on the currently selected NPU device."""

    def __init__(self, compiled):
        module = compiled.jit_module
        config = module.print_config
        if (module.ub_dynamic_base >= 0 or config.uses_scalar_print or config.uses_tensor_print
                or config.simt_print_formats is not None or config.print_argument_layout):
            raise RuntimeError("CATLASS DSL queued launch requires static UB and no device printing")
        with _launcher_lock:
            self._launch = _load_launcher().launch
        # The pinned CATLASS DSL runtime lazily loads the binary per device.
        self._executor = compiled._get_executor()
        self._pack = compiled.execution_args.generate_launch_payload
        self._name = compiled.entrypoint

    def __call__(self, arguments, tensors, *, block_num):
        self._launch(self._name, self._executor.function_handle, block_num, self._pack(arguments), tensors)
