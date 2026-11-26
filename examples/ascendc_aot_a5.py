"""Execute an exported standard op-package via ACLNN, optionally denying exec.

The fixed 16x16 FP32 probe submits via the packaged Torch/ACLNN bridge and
checks all 256 output values against an independent CPU GEMM.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path


def deny_compilers(cann_root):
    """Linux seccomp: deny *all* execve/execveat, including absolute compilers.

    This applies only to this test process and its children, needs no privilege,
    and changes neither the SDK nor the system's mount/filesystem state.
    """
    import errno
    import shutil
    import subprocess

    os.environ["PATH"] = "/usr/bin:/bin"
    assert shutil.which("bisheng") is None and shutil.which("ccec") is None
    seccomp = C.CDLL("libseccomp.so.2")
    seccomp.seccomp_init.argtypes, seccomp.seccomp_init.restype = [C.c_uint32], C.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = C.c_int
    seccomp.seccomp_rule_add.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint]
    seccomp.seccomp_rule_add.restype = C.c_int
    seccomp.seccomp_load.argtypes, seccomp.seccomp_load.restype = [C.c_void_p], C.c_int
    seccomp.seccomp_release.argtypes, seccomp.seccomp_release.restype = [C.c_void_p], None
    context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for syscall in (b"execve", b"execveat"):
            number = seccomp.seccomp_syscall_resolve_name(syscall)
            if number < 0 or seccomp.seccomp_rule_add(context, 0x50000 | errno.EPERM, number, 0):
                raise RuntimeError("could not deny " + syscall.decode())
        if seccomp.seccomp_load(context):
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)
    try:
        subprocess.run([str(cann_root / "bin/bisheng"), "--version"], check=True, capture_output=True)
    except PermissionError as error:
        assert error.errno == errno.EPERM
        print("COMPILER_BLOCKED: absolute bisheng --version -> PermissionError errno=1", flush=True)
    else:
        raise AssertionError("compiler was executable under the guard")

    def forbidden(*args, **kwargs):
        raise AssertionError("AOT runtime attempted compiler discovery/compilation/subprocess")

    subprocess.Popen = forbidden
    from flashinfer.jit import AscendCBackend, AscendCToolchain

    AscendCToolchain.discover = forbidden
    AscendCBackend.compile = forbidden
    print("COMPILER_GUARD: PATH bisheng=None ccec=None; execve/execveat=EPERM; discovery/JIT=forbidden", flush=True)


def execute(module, cann_root):
    import torch
    from flashinfer.jit.tensor_adapter import TorchOpPackageRunner

    runner = TorchOpPackageRunner(module, module.recipe)
    a = (torch.arange(256) % 7 - 3).float().reshape(16, 16)
    b = (torch.arange(256) % 5 - 2).float().reshape(16, 16)
    out = torch.empty_like(a, device="npu")
    runner(a.npu(), b.npu(), out)
    result = out.cpu()
    expected = a @ b
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    return dict(shape=[16, 16, 16], device=out.device.index,
                max_abs_error=(result - expected).abs().max().item(), first8=result.flatten()[:8].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="verified exported generation directory")
    parser.add_argument("--soc", default="Ascend950")
    parser.add_argument("--cann-root", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--deny-compilers", action="store_true")
    args = parser.parse_args()
    import torch_npu
    if args.deny_compilers:
        deny_compilers(args.cann_root)
    from flashinfer.jit.op_package import load_op_package

    from flashinfer.runtime.targets import canonical_soc_model
    module = load_op_package(args.package, target_soc=canonical_soc_model(args.soc))
    print("LOAD_OK: ascendc_op_package entry=" + module.recipe.host_entry, flush=True)
    result = execute(module, args.cann_root)
    print("PASS", json.dumps(dict(implementation=module.recipe.implementation,
        compilers_blocked=args.deny_compilers, **result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
