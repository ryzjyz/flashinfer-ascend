"""Prepare per-kernel acceptance releases through the framework AOT builder."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def build_release(kernel_id, root):
    from flashinfer.aot.build import build_production_packages

    root = Path(root)
    return build_production_packages(
        output_dir=root / "release",
        cache_dir=root / "build-cache",
        kernel_ids=(kernel_id,),
    )[0]


def installed_release(release, directory):
    """Stage the real wheel layout privately with setuptools' production build command."""
    environment = dict(os.environ, FLASHINFER_AOT_PREBUILT=str(release))
    for key in ("FLASHINFER_AOT_ALL", "FLASHINFER_AOT_CONFIG", "FLASHINFER_AOT_SOCS"):
        environment.pop(key, None)
    subprocess.run([sys.executable, "-B", "setup.py", "--quiet", "egg_info", "--egg-base", str(directory),
                    "build_py", "--build-lib", str(directory)], cwd=ROOT, env=environment, check=True)
    (directory / "tests").symlink_to(ROOT / "tests", target_is_directory=True)


def runtime_only_cann(directory):
    original = Path(os.environ["ASCEND_HOME_PATH"])
    runtime = directory / "runtime-only-cann"
    runtime.mkdir()
    for name in ("lib64", "share", "conf", "opp", "runtime"):
        (runtime / name).symlink_to(original / name, target_is_directory=True)
    (runtime / "compiler").mkdir()  # torch_npu checks for this directory at import.
    assert not (runtime / "bin/bisheng").exists()
    return {"ASCEND_HOME_PATH": str(runtime), "PATH": "/usr/bin:/bin"}


def run_release_scenario(module, scenario, store_root, *, search_root=None, bundled_release=None,
                         env=None, python=sys.executable, timeout=240):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    if search_root is not None:
        environment["FLASHINFER_AOT_PATHS"] = str(search_root)
    cwd = ROOT
    if bundled_release is not None:
        cwd = store_root.parent / "installed"
        cwd.mkdir()
        installed_release(bundled_release, cwd)
        environment.pop("FLASHINFER_ACLNN_ADAPTER_PATH", None)
        environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    environment.update(env or {})
    command = [str(python), "-B", "-m", module, scenario, str(store_root)]
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"scenario"' in line)
    assert evidence["result"] == "PASS" and evidence["scenario"] == scenario
    print(json.dumps(evidence, sort_keys=True))
    return evidence


def scenario_main(run_scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("store_root", type=Path)
    args = parser.parse_args()
    run_scenario(args.scenario, args.store_root)
