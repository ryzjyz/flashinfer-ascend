"""CATLASS Attention installation, dispatch and build dependency checks on Ascend950."""

import os
from pathlib import Path

import pytest

from tests.support.aot_release import run_release_scenario, runtime_only_cann

pytestmark = pytest.mark.requires_soc("Ascend950")

from tests.support.flash_attention_acceptance import CATLASS_ROOT, PYTHON


@pytest.fixture(scope="module")
def flash_attention_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "requires real NPU; a skip is not T4 acceptance"
    assert torch.npu.device_count() == 8, "expected the declared eight Ascend devices"
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    assert (cann / "bin/bisheng").is_file(), "real bisheng is required to build the AOT fixture"
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from tests.support.aot_release import build_release

    original = os.environ.get("FLASHINFER_CATLASS_ROOT")
    os.environ["FLASHINFER_CATLASS_ROOT"] = str(CATLASS_ROOT)
    try:
        return build_release(FLASH_ATTENTION_KERNEL_ID, tmp_path_factory.mktemp("flash-attention-release"))
    finally:
        if original is None:
            os.environ.pop("FLASHINFER_CATLASS_ROOT", None)
        else:
            os.environ["FLASHINFER_CATLASS_ROOT"] = original


def execute(scenario, tmp_path, *, bundled_release=None):
    env = {"FLASHINFER_CATLASS_ROOT": str(CATLASS_ROOT), "FLASHINFER_TEST_CATLASS_ROOT": str(CATLASS_ROOT)}
    if scenario == "hit_disabled":
        env.update(runtime_only_cann(tmp_path))
        absent = tmp_path / "absent-build-catlass"
        assert not absent.exists()
        env["FLASHINFER_CATLASS_ROOT"] = str(absent)
    return run_release_scenario("tests.support.flash_attention_acceptance", scenario, tmp_path / "cold-store",
                                env=env, bundled_release=bundled_release, python=PYTHON, timeout=300)


@pytest.mark.real_compile
def test_flash_attention_aot_without_compiler(flash_attention_release, tmp_path):
    evidence = execute("hit_disabled", tmp_path, bundled_release=flash_attention_release)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0
    assert evidence["toolchain_calls"] == 0
    assert not evidence["compiler_path_visible"] and not evidence["compiler_home_visible"]
    assert not evidence["catlass_root_available"]


@pytest.mark.real_compile
def test_flash_attention_envelope_rejected_in_dispatch(flash_attention_release, tmp_path):
    evidence = execute("out_of_range", tmp_path)
    assert evidence["category"] == "dispatch" and not evidence["runner_resolved"]
    assert evidence["compiler_calls"] == 0 and len(evidence["rejected"]) == 13


@pytest.mark.real_compile
def test_flash_attention_missing_catlass_dependency(tmp_path):
    evidence = execute("missing_dependency", tmp_path)
    assert evidence["category"] == "dependency" and evidence["compiler_calls"] == 0
    assert evidence["stage"] == "build_source_resolution"
