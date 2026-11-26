"""Check immutable AOT generations and missing host symbols using the shipped Matmul."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from flashinfer._matmul import AOT_KERNEL_ID
from flashinfer.ascendc import load_contract
from flashinfer.jit import DiskJitBuildStore
from flashinfer.jit.discovery import AotPackageIndex
from flashinfer.operators.pytest_support import prepare_release


@pytest.fixture(scope="module")
def matmul_release(tmp_path_factory):
    env = prepare_release("matmul", tmp_path_factory.mktemp("runtime-aot"), kernel_id=AOT_KERNEL_ID)
    descriptor, _, spec = load_contract(AOT_KERNEL_ID)
    index = AotPackageIndex((Path(env["FLASHINFER_AOT_PATHS"]),))
    return env, index.find(spec, descriptor=descriptor, target_soc=spec.target_soc)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
@pytest.mark.parametrize("change", ["missing_symbol", "second_generation"])
def test_verified_release_errors_do_not_fall_back(matmul_release, tmp_path, change):
    env, match = matmul_release
    store = DiskJitBuildStore(tmp_path / "store")
    with store.staging(match.spec) as staging:
        shutil.copytree(match.directory, staging, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("manifest.json", "build.log"))
        if change == "missing_symbol":
            package = json.loads((staging / "op-package.json").read_text())
            library = staging / package["api_library"]
            symbol = match.recipe.host_entry.encode()
            data = library.read_bytes()
            assert symbol in data
            library.write_bytes(data.replace(symbol, b"x" + symbol[1:]))
        else:
            with (staging / "ascendc-recipe.json").open("ab") as receipt:
                receipt.write(b"\n")
        other = store.publish(match.spec, staging, match.record.artifact)
    assert other.fingerprint != match.record.fingerprint
    changed = store.directory(other)
    worker = dict(os.environ, **env, FLASHINFER_NPU_JIT_DISABLE="1")
    worker.pop("ASCEND_CUSTOM_OPP_PATH", None)
    worker["FLASHINFER_AOT_PATHS"] = str(changed if change == "missing_symbol" else match.directory)
    subprocess.run([sys.executable, "-B", "-m", __spec__.name, change, str(changed)],
                   cwd=Path(__file__).resolve().parents[2], env=worker, check=True, timeout=180)


def check_release(change, changed):
    from unittest.mock import patch
    import torch
    import torch_npu
    import flashinfer
    from flashinfer.jit import AscendCBackend, AscendCToolchain
    from flashinfer.jit.discovery import AotLoadError

    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    runner = flashinfer.get_global_registry().get(AOT_KERNEL_ID).runner
    runner.index = AotPackageIndex((Path(os.environ["FLASHINFER_AOT_PATHS"]),))
    x = torch.ones(16, 16, device="npu", dtype=torch.bfloat16)
    y = x + 1
    torch.testing.assert_close(y.cpu(), torch.full((16, 16), 2, dtype=torch.bfloat16))

    def matmul():
        return flashinfer.matmul(x, y, backend="ascendc", kernel_id=AOT_KERNEL_ID)

    forbidden = AssertionError("AOT activation must not invoke a compiler")
    with patch.object(AscendCBackend, "compile", side_effect=forbidden), \
         patch.object(AscendCToolchain, "discover", side_effect=forbidden):
        if change == "missing_symbol":
            with pytest.raises(AotLoadError, match="no ACLNN symbol") as error:
                matmul()
        else:
            expected = x.cpu() @ y.cpu()
            torch.testing.assert_close(matmul().cpu(), expected, atol=0, rtol=0)
            descriptor, _, spec = load_contract(AOT_KERNEL_ID)
            match = AotPackageIndex((Path(changed),)).find(spec, descriptor=descriptor, target_soc=spec.target_soc)
            with pytest.raises(AotLoadError, match="process-scoped.*fresh process") as error:
                match.load()
            torch.testing.assert_close(matmul().cpu(), expected, atol=0, rtol=0)
        assert error.value.__cause__ is not None


if __name__ == "__main__":
    check_release(*sys.argv[1:])
