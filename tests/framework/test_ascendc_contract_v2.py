"""Current ACLNN recipe, kernel metadata and native binding contracts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from flashinfer.jit import load_build_recipes
from flashinfer.jit.core import JitSpec
from flashinfer.jit.op_package import OP_PACKAGE_ADAPTER_HASH, verify_op_package
from flashinfer.jit.recipe import BuildRecipe, ShapeConstraint, TensorParameter
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.runtime import SchemaError

from tests.packaging.test_ascendc_op_package import elf_fixture, write_binding, write_fixture


@pytest.fixture
def v2():
    root = Path(__file__).resolve().parents[2]
    probe = load_build_recipes(root / "csrc/recipes/matmul_probe.json")[0]
    return replace(probe,
        tiling_host_entry="FixtureTiling", workspace_host_entry="FixtureTiling",
        workspace_bytes="dynamic", task_type="aic_aiv", sync_policy="cross_core_flags",
        tensor_parameters=(TensorParameter("a", "input", "bfloat16", ("M", "K")),
                           TensorParameter("b", "input", "bfloat16", ("K", "N")),
                           TensorParameter("bias", "input", "bfloat16", ("N",)),
                           TensorParameter("out", "output", "bfloat16", ("M", "N"))),
        shape_constraints=tuple(ShapeConstraint(n, 16, 256, 16) for n in ("M", "N", "K")))


def metadata(m=16, n=32, k=64):
    return dict(shapes=dict(a=(m, k), b=(k, n), bias=(n,), out=(m, n)),
                dtypes={n: "bfloat16" for n in ("a", "b", "bias", "out")})


@pytest.mark.parametrize("change,message", [
    (dict(tiling_mode="fixed"), "host_dynamic"),
    (dict(tiling_schema_version=1), "schema >= 2"),
    (dict(tiling_host_entry=None), "tiling_host_entry"),
    (dict(tiling_host_entry="bad entry"), "tiling_host_entry"),
    (dict(workspace_host_entry=None), "workspace_host_entry"),
    (dict(workspace_host_entry="bad entry"), "workspace_host_entry"),
    (dict(workspace_bytes=-1), "workspace_bytes"),
    (dict(workspace_bytes=True), "workspace_bytes"),
    (dict(workspace_bytes="unknown"), "workspace_bytes"),
    (dict(sync_policy="compiler_default"), "synchronization"),
    (dict(task_type="aiv_only"), "task_type"),
    (dict(tensor_parameters=()), "tensor_parameters"),
    (dict(shape_constraints=()), "every tensor shape symbol"),
    (dict(block_dim=0), "block_dim"),
])
def test_v2_rejects_incomplete_or_unimplemented_protocols(v2, change, message):
    with pytest.raises(SchemaError, match=message):
        replace(v2, **change)


def test_v2_requires_unique_ordered_tensor_arguments_and_exact_symbol_bounds(v2):
    for parameters in (v2.tensor_parameters + v2.tensor_parameters[:1],
                       v2.tensor_parameters[-1:] + v2.tensor_parameters[:-1]):
        with pytest.raises(SchemaError, match="tensor_parameters"):
            replace(v2, tensor_parameters=parameters)
    for constraints in (v2.shape_constraints + v2.shape_constraints[:1],
                        v2.shape_constraints + (ShapeConstraint("Unused", 1, 16),)):
        with pytest.raises(SchemaError, match="shape_constraints"):
            replace(v2, shape_constraints=constraints)


@pytest.mark.parametrize("argument", [
    dict(name="workspace", role="input", dtype="float32", shape=(16,)),
    dict(name="a", role="scalar", dtype="float32", shape=(16,)),
    dict(name="a", role="input", dtype="opaque", shape=(16,)),
    dict(name="a", role="input", dtype="float32", shape=(True,)),
    dict(name="a", role="input", dtype="float32", shape=(0,)),
])
def test_tensor_parameters_are_typed_tensors_not_unchecked_foreign_arguments(argument):
    with pytest.raises(SchemaError):
        TensorParameter(**argument)


@pytest.mark.parametrize("bounds", [(0, 16, 1), (16, 8, 1), (1, 16, 0), (1, 3, 4), (True, 16, 1)])
def test_shape_bounds_must_describe_a_nonempty_positive_integer_range(bounds):
    with pytest.raises(SchemaError, match="range"):
        ShapeConstraint("M", *bounds)


def test_v2_contract_is_round_trippable_and_identity_binds_every_new_field(v2):
    assert BuildRecipe.from_dict(json.loads(json.dumps(v2.to_dict()))) == v2
    assert v2.to_dict()["schema_version"] == 2
    assert v2.device_parameter_order == ("a", "b", "bias", "out", "workspace", "tiling")
    for change in (dict(tiling_schema_version=3), dict(tiling_host_entry="OtherTiling"),
                   dict(workspace_host_entry="OtherWorkspace"),
                   dict(workspace_bytes=4096, workspace_host_entry=None),
                   dict(task_type="aic_only", sync_policy="compiler_default"),
                   dict(tensor_parameters=(replace(v2.tensor_parameters[0], dtype="float32"),)
                        + v2.tensor_parameters[1:]),
                   dict(shape_constraints=(ShapeConstraint("M", 16, 128, 16),)
                        + v2.shape_constraints[1:])):
        assert replace(v2, **change).fingerprint != v2.fingerprint
    assert replace(v2, workspace_bytes=0, workspace_host_entry=None).workspace_bytes == 0


def test_dispatch_metadata_validation_checks_cross_tensor_dimensions_dtype_and_bounds(v2):
    assert v2.validate_tensor_metadata(**metadata()) == dict(M=16, K=64, N=32)
    assert v2.validate_tensor_metadata(**metadata(256, 256, 256))["M"] == 256
    for unsupported in (metadata(17), metadata(272), metadata(0)):
        with pytest.raises(SchemaError, match="dispatch shape"):
            v2.validate_tensor_metadata(**unsupported)
    inconsistent = metadata()
    inconsistent["shapes"]["bias"] = (16,)
    with pytest.raises(SchemaError, match="symbol N is inconsistent"):
        v2.validate_tensor_metadata(**inconsistent)
    rank = metadata()
    rank["shapes"]["bias"] = (1, 32)
    with pytest.raises(SchemaError, match="requires rank 1"):
        v2.validate_tensor_metadata(**rank)
    dtype = metadata()
    dtype["dtypes"]["bias"] = "float32"
    with pytest.raises(SchemaError, match="dtype for bias must be bfloat16"):
        v2.validate_tensor_metadata(**dtype)
    missing = metadata()
    missing["shapes"].pop("bias")
    with pytest.raises(SchemaError, match="match recipe parameters"):
        v2.validate_tensor_metadata(**missing)


def test_dtype_signatures_preserve_old_serialization_and_bind_new_identity(v2):
    old = v2.to_dict()
    assert "dtype_signatures" not in old
    assert v2.tensor_dtype_signatures == (("bfloat16",) * 4,)
    extended = replace(v2, dtype_signatures=(("bfloat16",) * 4, ("float16",) * 4))
    assert extended.fingerprint != v2.fingerprint
    assert BuildRecipe.from_dict(json.loads(json.dumps(extended.to_dict()))) == extended
    for dtype in ("bfloat16", "float16"):
        data = metadata()
        data["dtypes"] = dict.fromkeys(extended.tensor_parameter_names, dtype)
        assert extended.validate_tensor_metadata(**data) == dict(M=16, K=64, N=32)
    data["dtypes"]["bias"] = "bfloat16"
    with pytest.raises(SchemaError, match="dtype signature"):
        extended.validate_tensor_metadata(**data)


def test_dtype_signatures_express_exact_mixed_combinations_without_cartesian_expansion(v2):
    mixed = ("float16", "bfloat16", "float32", "float16")
    extended = replace(v2, dtype_signatures=(("bfloat16",) * 4, mixed))
    data = metadata()
    data["dtypes"] = dict(zip(extended.tensor_parameter_names, mixed))
    assert extended.validate_tensor_metadata(**data)["N"] == 32
    data["dtypes"]["bias"] = "bfloat16"
    with pytest.raises(SchemaError, match="dtype signature"):
        extended.validate_tensor_metadata(**data)


@pytest.mark.parametrize("signatures", [
    None, "float16", ["float16"], [("bfloat16",) * 3],
    [("bfloat16",) * 4, ("bfloat16",) * 4], [("float16",) * 4],
    [("bfloat16",) * 4, ("opaque",) * 4], [("bfloat16",) * 4, ([],) * 4],
])
def test_dtype_signatures_reject_malformed_or_ambiguous_combinations(v2, signatures):
    with pytest.raises(SchemaError, match="dtype_signatures"):
        replace(v2, dtype_signatures=signatures)


@pytest.fixture
def mixed_package(v2, tmp_path):
    spec = JitSpec(name="mixed_fixture", domain="test", generator_id="ascendc", generator_version="1",
        target_soc=v2.target_soc, environment_fingerprint="a" * 64, specialization_fingerprint="b" * 64,
        input_fingerprints=(("build_recipe", v2.fingerprint), ("output_adapter", OP_PACKAGE_ADAPTER_HASH)),
        kernel_id=v2.kernel_id, source=v2.source, output_path="op-package.json", entry_points=(v2.host_entry,))
    paths = write_fixture(tmp_path, v2, spec)
    metadata_path = tmp_path / paths["metadata"]
    metadata = json.loads(metadata_path.read_text())
    entry = metadata["kernelList"][0]["kernelName"]
    binary = elf_fixture((entry + "_mix_aic", entry + "_mix_aiv"), machine=0x1029, kind=2)
    (tmp_path / paths["binary"]).write_bytes(binary)
    metadata.pop("core_type")
    metadata.update(coreType="MIX", magic="RT_DEV_BINARY_MAGIC_ELF", intercoreSync=1,
                    sha256=hashlib.sha256(binary).hexdigest(), workspace=dict(num=1, size=[-1]),
                    supportInfo=dict(opMode="dynamic", inputs=[dict(name=p.name, dtype=p.dtype)
                        for p in v2.tensor_parameters if p.role == "input"],
                        outputs=[dict(name=p.name, dtype=p.dtype)
                        for p in v2.tensor_parameters if p.role == "output"]))
    metadata["kernelList"][0].update(kernelType="MIX_AIC", crossCoreSync=1, taskRation="1:2")
    metadata_path.write_text(json.dumps(metadata))
    return tmp_path, spec, v2, paths, metadata


def test_mixed_package_verifies_both_core_entries_and_declared_abi(mixed_package):
    directory, spec, recipe, _, _ = mixed_package
    assert verify_op_package(directory, spec)[1] == recipe


def test_aicpu_package_checks_registration_and_device_elf(mixed_package):
    from flashinfer.jit.op_package import OpPackageSymbolError
    from flashinfer.runtime import SourceRef

    directory, spec, recipe, _, _ = mixed_package
    cmake = SourceRef("operator", "fixture/CMakeLists.txt", "c" * 64)
    recipe = replace(recipe, task_type="aicpu", device_entry="RunCpuKernel", sync_policy="compiler_default",
                     tiling_mode="none", tiling_host_entry=None, allow_jit_fallback=False,
                     workspace_host_entry=recipe.host_entry + "GetWorkspaceSize",
                     cmake_entry=cmake.locator, extra_build_files=(cmake,))
    assert recipe.device_parameter_order == ()
    with pytest.raises(SchemaError, match="standalone native AOT"):
        replace(recipe, tiling_mode="host_dynamic", tiling_host_entry="FixtureTiling")
    with pytest.raises(SchemaError, match="GetWorkspaceSize"):
        replace(recipe, workspace_host_entry="AnotherWorkspace")

    spec = replace(spec, input_fingerprints=tuple(
        (key, recipe.fingerprint if key == "build_recipe" else value) for key, value in spec.input_fingerprints))
    package = directory / "aicpu"
    write_fixture(package, recipe, spec)
    vendor = package / "packages/vendors" / recipe.package_vendor
    cpu = vendor / "op_impl/cpu"
    registration = cpu / "config/cust_aicpu_kernel.json"
    registration.parent.mkdir(parents=True)
    binary = cpu / "aicpu_kernel/impl/libfixture.so"
    binary.parent.mkdir(parents=True)
    info = dict(engine="DNN_VM_AICPU", opKernelLib="CUSTAICPUKernel", userDefined="True",
                functionName=recipe.device_entry, kernelSo=binary.name)
    registration.write_text(json.dumps({recipe.package_op_type: {"opInfo": info}}))
    binary.write_bytes(elf_fixture((recipe.device_entry,), machine=183, kind=3))
    assert verify_op_package(package, spec)[1] == recipe
    binary.write_bytes(elf_fixture(("other_entry",), machine=183, kind=3))
    with pytest.raises(OpPackageSymbolError, match="no device entry"):
        verify_op_package(package, spec)
    info["kernelSo"] = "../libfixture.so"
    registration.write_text(json.dumps({recipe.package_op_type: {"opInfo": info}}))
    with pytest.raises(SchemaError, match="invalid custom AICPU kernel registration"):
        verify_op_package(package, spec)


def test_vector_package_checks_aiv_metadata_and_unsuffixed_entry(mixed_package):
    directory, spec, recipe, paths, metadata = mixed_package
    recipe = replace(recipe, task_type="aiv_only", sync_policy="compiler_default")
    with pytest.raises(SchemaError, match="task_type/sync_policy"):
        replace(recipe, sync_policy="cross_core_flags")
    spec = replace(spec, input_fingerprints=tuple(
        (key, recipe.fingerprint if key == "build_recipe" else value) for key, value in spec.input_fingerprints))
    paths = write_fixture(directory, recipe, spec)
    record = directory / paths["metadata"]
    vector = dict(json.loads(record.read_text()), core_type="AIV", coreType="VectorCore",
                  magic="RT_DEV_BINARY_MAGIC_ELF_AIVEC", supportInfo=metadata["supportInfo"], workspace=metadata["workspace"])
    record.write_text(json.dumps(vector))
    assert verify_op_package(directory, spec)[1] == recipe
    for key, value in (("core_type", "AIC"), ("coreType", "MIX"), ("magic", "RT_DEV_BINARY_MAGIC_ELF_AICUBE")):
        record.write_text(json.dumps(dict(vector, **{key: value})))
        with pytest.raises(SchemaError, match="AIV executable"):
            verify_op_package(directory, spec)
    record.write_text(json.dumps(dict(vector, kernelList=[{"kernelName": recipe.package_op_type + "_missing"}])))
    with pytest.raises(SchemaError, match="does not define"):
        verify_op_package(directory, spec)


def test_synchronized_vector_package_requires_aiv_entry_and_ffts_metadata(mixed_package):
    directory, spec, recipe, paths, metadata = mixed_package
    recipe = replace(recipe, task_type="aiv_hard_sync")
    with pytest.raises(SchemaError, match="synchronization"):
        replace(recipe, sync_policy="compiler_default")
    spec = replace(spec, input_fingerprints=tuple(
        (k, recipe.fingerprint if k == "build_recipe" else v) for k, v in spec.input_fingerprints))
    paths = write_fixture(directory, recipe, spec)
    kernel = dict(metadata["kernelList"][0], taskRation="0:1")
    binary = elf_fixture((kernel["kernelName"] + "_mix_aiv",), machine=0x1029, kind=2)
    (directory / paths["binary"]).write_bytes(binary)
    record = directory / paths["metadata"]
    metadata.update(kernelList=[kernel], sha256=hashlib.sha256(binary).hexdigest())
    record.write_text(json.dumps(metadata))
    assert verify_op_package(directory, spec)[1] == recipe
    for change in (dict(intercoreSync=0), dict(kernelList=[dict(kernel, crossCoreSync=0)]),
                   dict(kernelList=[dict(kernel, taskRation="1:2")]), dict(coreType="VectorCore")):
        record.write_text(json.dumps(dict(metadata, **change)))
        with pytest.raises(SchemaError, match="synchronized AIV"):
            verify_op_package(directory, spec)
    record.write_text(json.dumps(dict(metadata, kernelList=[dict(kernel, kernelName=kernel["kernelName"] + "_missing")])))
    with pytest.raises(SchemaError, match="does not define"):
        verify_op_package(directory, spec)


@pytest.fixture
def string_attribute_package(mixed_package):
    directory, spec, recipe, paths, metadata = mixed_package
    recipe = replace(recipe, attributes=(AttributeParameter("layout", "string", "BSND"),
                                         AttributeParameter("enabled", "bool", False)))
    spec = replace(spec, input_fingerprints=tuple(
        (key, recipe.fingerprint if key == "build_recipe" else value)
        for key, value in spec.input_fingerprints))
    manifest_path = directory / spec.output_path
    manifest = json.loads(manifest_path.read_text())
    write_binding(directory, recipe)
    manifest.update(recipe=recipe.to_dict(), spec_fingerprint=spec.fingerprint)
    manifest_path.write_text(json.dumps(manifest))
    for key in ("inputs", "outputs"):
        for parameter in metadata["supportInfo"][key]:
            parameter.update(format="ND", paramType="required")
    metadata["supportInfo"]["attrs"] = [dict(name="layout", dtype="str"), dict(name="enabled", dtype="bool")]
    metadata_path = directory / paths["metadata"]
    metadata_path.write_text(json.dumps(metadata))
    return directory, spec, recipe, metadata_path, metadata


def test_string_attributes_match_the_exact_cann_contract(string_attribute_package):
    directory, spec, recipe, _, _ = string_attribute_package
    assert verify_op_package(directory, spec)[1] == recipe


@pytest.mark.parametrize("mutation", ["type", "name", "order"])
def test_string_attributes_keep_exact_type_name_and_order_checks(string_attribute_package, mutation):
    directory, spec, _, path, metadata = string_attribute_package
    attrs = metadata["supportInfo"]["attrs"]
    if mutation == "order":
        attrs.reverse()
    else:
        attrs[0]["dtype" if mutation == "type" else "name"] = "int" if mutation == "type" else "wrong"
    path.write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match="attribute order/types"):
        verify_op_package(directory, spec)


def test_absent_optional_tensor_resolves_only_a_coherent_declared_dtype(v2):
    recipe = replace(v2,
        tensor_parameters=tuple(replace(p, optional=True) if p.name == "bias" else p
                                for p in v2.tensor_parameters),
        dtype_signatures=(("bfloat16",) * 4, ("float16",) * 4))
    data = metadata()
    data["shapes"]["bias"] = None
    data["dtypes"] = dict(a="float16", b="float16", bias=None, out="float16")
    assert recipe.validate_tensor_metadata(**data) == dict(M=16, K=64, N=32)
    data["dtypes"]["b"] = "bfloat16"
    with pytest.raises(SchemaError, match="not a declared combination"):
        recipe.validate_tensor_metadata(**data)
    data["dtypes"]["b"] = "float16"
    data["dtypes"]["bias"] = "float16"
    with pytest.raises(SchemaError, match="absent dtype"):
        recipe.validate_tensor_metadata(**data)


@pytest.fixture
def dual_dtype_package(mixed_package):
    directory, spec, recipe, paths, metadata = mixed_package
    recipe = replace(recipe, dtype_signatures=(("bfloat16",) * 4, ("float16",) * 4))
    spec = replace(spec, input_fingerprints=tuple(
        (key, recipe.fingerprint if key == "build_recipe" else value)
        for key, value in spec.input_fingerprints))
    manifest_path = directory / spec.output_path
    manifest = json.loads(manifest_path.read_text())
    write_binding(directory, recipe)
    manifest.update(recipe=recipe.to_dict(), spec_fingerprint=spec.fingerprint)
    manifest_path.write_text(json.dumps(manifest))
    other_metadata = json.loads(json.dumps(metadata))
    for key in ("inputs", "outputs"):
        for parameter in other_metadata["supportInfo"][key]:
            parameter["dtype"] = "float16"
    other_path = directory / paths["metadata"].replace("kernel.json", "float16.json")
    other_path.write_text(json.dumps(other_metadata))
    config_path = directory / paths["config"]
    config = json.loads(config_path.read_text())
    original = config[recipe.package_op_type]["binaryList"][0]
    config[recipe.package_op_type]["binaryList"].append(dict(
        original, jsonPath=original["jsonPath"].replace("kernel.json", "float16.json")))
    config_path.write_text(json.dumps(config))
    return directory, spec, recipe, config_path, config, other_path, other_metadata


def test_package_requires_executables_for_every_declared_dtype_combination(dual_dtype_package):
    directory, spec, recipe, *_ = dual_dtype_package
    assert verify_op_package(directory, spec)[1] == recipe


@pytest.mark.parametrize("missing", [0, 1])
def test_package_rejects_either_missing_dtype_variant(dual_dtype_package, missing):
    directory, spec, recipe, config_path, config, *_ = dual_dtype_package
    config[recipe.package_op_type]["binaryList"].pop(missing)
    config_path.write_text(json.dumps(config))
    with pytest.raises(SchemaError, match="missing a declared dtype signature"):
        verify_op_package(directory, spec)


def test_package_rejects_mixed_dtype_even_when_each_dtype_is_allowed(dual_dtype_package):
    directory, spec, _, _, _, metadata_path, metadata = dual_dtype_package
    metadata["supportInfo"]["inputs"][0]["dtype"] = "bfloat16"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match="order/types"):
        verify_op_package(directory, spec)


@pytest.mark.parametrize("change,message", [
    (dict(coreType="AiCore"), "AIC\\+AIV"),
    (dict(magic="RT_DEV_BINARY_MAGIC_ELF_AICUBE"), "AIC\\+AIV"),
    (dict(intercoreSync=0), "synchronized"),
    (dict(workspace=dict(num=1, size=[0])), "dynamic workspace"),
    (dict(supportInfo=dict(opMode="static")), "dynamic host tiling"),
])
def test_mixed_package_rejects_metadata_that_weakens_v2(mixed_package, change, message):
    directory, spec, _, paths, metadata = mixed_package
    metadata.update(change)
    (directory / paths["metadata"]).write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match=message):
        verify_op_package(directory, spec)


def test_mixed_package_rejects_missing_vector_entry(mixed_package):
    directory, spec, _, paths, metadata = mixed_package
    entry = metadata["kernelList"][0]["kernelName"]
    (directory / paths["binary"]).write_bytes(elf_fixture((entry + "_mix_aic",), machine=0x1029, kind=2))
    with pytest.raises(SchemaError, match="metadata entries"):
        verify_op_package(directory, spec)


@pytest.mark.parametrize("invalid", ["order", "dtype", "cross_core"])
def test_mixed_package_rejects_tensor_abi_or_cross_core_mismatch(mixed_package, invalid):
    directory, spec, _, paths, metadata = mixed_package
    if invalid == "order":
        metadata["supportInfo"]["inputs"].reverse()
    elif invalid == "dtype":
        metadata["supportInfo"]["inputs"][0]["dtype"] = "float32"
    else:
        metadata["kernelList"][0]["crossCoreSync"] = 0
    (directory / paths["metadata"]).write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match="order/types|synchronized"):
        verify_op_package(directory, spec)
