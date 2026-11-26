"""Independent operators coexist without permitting registration replacement."""

import ctypes as C
from dataclasses import replace
import os
from pathlib import Path

import pytest

from flashinfer.jit import DiskJitBuildStore
from flashinfer.jit.build import _verify_output
from flashinfer.jit import op_package as package
from flashinfer.runtime import SchemaError
from tests.packaging.test_ascendc_op_package import fake_runtime, generation, unit, write_fixture


def publish_other(unit, root, **changes):
    recipe, _, spec = unit
    recipe = replace(recipe, **dict(dict(package_vendor="flashinfer_other_operator",
                                         package_op_type="FiOtherOperator",
                                         host_entry="aclnnFiOtherOperator"), **changes))
    fingerprints = dict(spec.input_fingerprints, build_recipe=recipe.fingerprint)
    spec = replace(spec, name="other_operator", entry_points=(recipe.host_entry,),
                   input_fingerprints=tuple(fingerprints.items()))
    store = DiskJitBuildStore(root)
    with store.staging(spec) as staging:
        write_fixture(staging, recipe, spec)
        record = store.publish(spec, staging, _verify_output(spec, staging))
    return store.directory(record)


def test_distinct_operators_retain_both_modules_and_verified_vendor_paths(unit, generation, fake_runtime, tmp_path):
    store, record, _ = generation
    first_path = store.directory(record)
    second_path = publish_other(unit, tmp_path / "second")
    first, second = package.load_op_packages((first_path, second_path), target_soc=record.artifact.target_soc)
    assert first is not second
    assert package.load_op_package(first_path, target_soc=record.artifact.target_soc) is first
    assert package.load_op_package(second_path, target_soc=record.artifact.target_soc) is second
    paths = [Path(path) for path in os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)]
    assert len(paths) == 2
    assert paths[0].samefile(first.vendor_path) and paths[1].samefile(second.vendor_path)
    assert len(fake_runtime) == 2


@pytest.mark.parametrize("collision", ["vendor", "operator"])
def test_existing_vendor_op_type_or_api_cannot_be_rebound(unit, generation, fake_runtime, tmp_path, collision):
    store, record, _ = generation
    module = package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc)
    previous = os.environ["ASCEND_CUSTOM_OPP_PATH"]
    # OpDef requires the ACLNN entry to match the op type; preserve that
    # contract while testing a registration collision across distinct vendors.
    changes = (dict(package_vendor=unit[0].package_vendor) if collision == "vendor" else
               dict(package_op_type=unit[0].package_op_type, host_entry=unit[0].host_entry))
    second = publish_other(unit, tmp_path / "second", **changes)
    with pytest.raises(SchemaError, match="fresh process"):
        package.load_op_package(second, target_soc=record.artifact.target_soc)
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == previous
    assert package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc) is module
    assert len(fake_runtime) == 1


def test_late_operator_requires_a_new_activation_group(unit, generation, fake_runtime, tmp_path):
    store, record, _ = generation
    first = package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc)
    previous = os.environ["ASCEND_CUSTOM_OPP_PATH"]
    second = publish_other(unit, tmp_path / "second")

    with pytest.raises(SchemaError, match="prepare all operators together in a fresh process"):
        package.load_op_package(second, target_soc=record.artifact.target_soc)
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == previous
    assert package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc) is first
    assert len(fake_runtime) == 1


def test_failed_group_activation_releases_all_aliases(unit, generation, fake_runtime, tmp_path, monkeypatch):
    store, record, _ = generation
    second = publish_other(unit, tmp_path / "second")
    original = C.CDLL
    aliases = []

    def reject_second(path, mode):
        aliases.extend(int(Path(part).parts[4]) for part in
                       os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep))
        if "flashinfer_other_operator" in str(path):
            raise OSError("second operator fixture dlopen failure")
        return original(path, mode)

    monkeypatch.setattr(C, "CDLL", reject_second)
    with pytest.raises(OSError, match="second operator fixture"):
        package.load_op_packages((store.directory(record), second), target_soc=record.artifact.target_soc)
    assert "ASCEND_CUSTOM_OPP_PATH" not in os.environ
    assert package._active_package is None
    assert len(set(aliases)) == 2
    for fd in set(aliases):
        with pytest.raises(OSError):
            os.fstat(fd)
