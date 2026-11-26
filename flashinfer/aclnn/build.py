"""Explicit build of the Torch/ACLNN host adapter; never called by import/run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def torch_abi():
    import torch
    import torch_npu

    return dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                cxx11_abi=torch.compiled_with_cxx11_abi())


def bundle_adapter(output_dir, *, directory=None):
    """Copy a verified, compatible prebuilt bridge without invoking a compiler."""
    from ._native import verified_adapter

    library, manifest = verified_adapter(directory)
    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/aclnn_torch_adapter/adapter.cpp"
    if not source.is_file():
        source = Path(__file__).resolve().parents[1] / "data/build/csrc/aclnn_torch_adapter/adapter.cpp"
    if json.loads(manifest.read_text()).get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Prebuilt ACLNN Torch adapter source identity mismatch; rebuild the adapter")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for path in (library, manifest):
        shutil.copyfile(path, destination / path.name)
    return tuple(destination / path.name for path in (library, manifest))


def _compile(source, library, includes, extra_sources=(), *, cann_root=None, compiler=None):
    import sysconfig
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    cann = Path(cann_root or os.environ["ASCEND_HOME_PATH"]).resolve(strict=True)
    npu = Path(torch_npu.__file__).resolve().parent
    compiler = compiler or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Building the ACLNN Torch adapter requires c++")
    command = [compiler, "-std=c++17", "-O2", "-g0", "-fPIC", "-fvisibility=hidden", "-shared",
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi())),
               str(source), *map(str, extra_sources), "-o", str(library)]
    for include in [*includes, *include_paths(), npu/"include", cann/"include", sysconfig.get_path("include")]:
        command += ["-I", str(include)]
    for directory in [*library_paths(), npu/"lib", cann/"lib64"]:
        command += ["-L", str(directory)]
    command += ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu", "-lascendcl", "-ldl",
                "-Wl,-rpath,$ORIGIN/../../../torch/lib:$ORIGIN/../../../torch_npu/lib"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("ACLNN adapter compilation failed:\n" + result.stdout)


def build_package_call(source, destination, source_root, *, cann_root=None, compiler=None):
    headers = Path(source_root) / "csrc/aclnn_torch_adapter"
    _compile(source, destination, [headers], [headers / "NPUBridge.cpp"], cann_root=cann_root, compiler=compiler)


def build_adapter(output_dir, *, source_root=None):
    import torch
    import torch_npu
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    source = root / "csrc/aclnn_torch_adapter/adapter.cpp"
    if not source.is_file():
        root = Path(__file__).resolve().parents[1] / "data/build"
        source = root / "csrc/aclnn_torch_adapter/adapter.cpp"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aclnn-build-", dir=destination) as temporary:
        library = Path(temporary) / "libflashinfer_aclnn.so"
        _compile(source, library, [source.parent])
        metadata = dict(schema_version=1, library=library.name, submission="torch_npu_task_queue",
                        sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        call_header_sha256=hashlib.sha256((source.parent/"call.h").read_bytes()).hexdigest())
        manifest = Path(temporary) / "manifest.json"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        # Rebuilding must not overwrite an inode mapped by an existing process.
        os.replace(library, destination / library.name)
        os.replace(manifest, destination / manifest.name)
    return tuple(destination / name for name in ("libflashinfer_aclnn.so", "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    for path in build_adapter(options.output_dir):
        print(path)
