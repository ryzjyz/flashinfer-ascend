"""Public op-package verification and process-scoped vendor loading.

Real compiler-free NPU execution is exercised by examples/ascendc_aot_a5.py.
"""

import ctypes as C
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import platform
import struct
import os
import subprocess
from types import SimpleNamespace

import pytest

from flashinfer.aot.build import _for_target
from flashinfer.jit import (
    AscendCBackend, AscendCToolchain, DiskJitBuildStore,
    JitCacheIndex, JitCompilationPolicy, JitSpecRegistry, build_and_load,
    declare_kernel,
)
from flashinfer.jit.build import _build_record, _verify_output
from flashinfer.jit.cache import verify_build_directory
from flashinfer.jit import op_package as package
from flashinfer.runtime import (
    ArtifactFormat, Backend, CompileMode, KernelDescriptor, KernelLaunchABI, KernelRegistry, SchemaError,
)
from flashinfer.aot.build import export_op_package
from flashinfer.jit import AscendCBuildConfig, BuildRecipe, load_build_recipes, resolve_build_recipe

ROOT = Path(__file__).resolve().parents[2]
TEST_TORCH_ABI = {"torch": "fixture", "torch_npu": "fixture"}


def elf_fixture(symbols, *, machine, kind):
    """Minimal ELF section/symbol data for verification, not executable code."""
    strings = b"\0"
    symbol_data = bytes(24)
    for name in symbols:
        offset = len(strings)
        strings += name.encode() + b"\0"
        symbol_data += struct.pack("<IBBHQQ", offset, 0x12, 0, 1, 0, 1)
    table_offset = 64 + 4 * 64
    string_offset = table_offset + len(symbol_data)
    header = b"\x7fELF\x02\x01\x01" + bytes(9)
    header += struct.pack("<HHIQQQIHHHHHH", kind, machine, 1, 0, 0, 64, 0, 64, 0, 0, 64, 4, 0)
    sections = bytes(64)
    sections += struct.pack("<IIQQQQIIQQ", 0, 1, 0, 0, 0, 0, 0, 0, 1, 0)
    sections += struct.pack("<IIQQQQIIQQ", 0, 11, 0, 0, table_offset, len(symbol_data), 3, 0, 8, 24)
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, string_offset, len(strings), 0, 0, 1, 0)
    return header + sections + symbol_data + strings


@pytest.fixture
def unit():
    config = AscendCBuildConfig(implementation="native", target_soc="Ascend950")
    recipe = config.select(load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json"), "matmul_probe")
    toolchain = AscendCToolchain(Path("/absent"), Path("/absent/bisheng"), Path("/absent/cmake"),
                                Path("/absent/c++"), (("fixture", "synthetic"),))
    resolved = resolve_build_recipe(recipe, source_root=ROOT, toolchain_fingerprint=toolchain.fingerprint)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
        compile_mode=CompileMode.AOT, source=recipe.source, target_soc=recipe.target_soc,
        tiling_schema_version=recipe.tiling_schema_version,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                  ("workspace", "workspace_bytes", "executor", "stream")))
    spec = declare_kernel(recipe.kernel_id, backend=backend,
        registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
        output_path="op-package.json", entry_point=recipe.host_entry,
        environment_fingerprint=toolchain.fingerprint, specialization_fingerprint="a" * 64)
    return recipe, backend, spec


def write_binding(directory, recipe):
    # These bytes exercise the package verifier. Code generation and real ELFs have separate tests.
    declaration = recipe.to_dict()
    record = dict(schema_version=1, kernel_id=recipe.kernel_id,
                  recipe_fingerprint=recipe.fingerprint, host_entry=recipe.host_entry,
                  tensor_parameters=declaration["tensor_parameters"], attributes=declaration.get("attributes", []))
    for suffix, field in ((".h", "header_sha256"), (".cpp", "source_sha256")):
        content = ("// Synthetic binding for " + recipe.host_entry + suffix).encode()
        (directory / ("fi_aclnn_adapter" + suffix)).write_bytes(content)
        record[field] = hashlib.sha256(content).hexdigest()
    (directory / "fi_aclnn_adapter.json").write_text(json.dumps(record))


def write_fixture(directory, recipe, spec):
    vendor = "packages/vendors/" + recipe.package_vendor
    kernel = vendor + "/op_impl/ai_core/tbe/kernel/"
    compute = package.package_target(recipe.target_soc)[0]
    entry = recipe.package_op_type + "_fixture_0"
    paths = dict(api=vendor + "/op_api/lib/libcust_opapi.so",
        binary=kernel + compute + "/probe/kernel.o", metadata=kernel + compute + "/probe/kernel.json",
        config=kernel + "config/" + compute + "/binary_info_config.json",
        tiling=vendor + "/op_impl/ai_core/tbe/op_tiling/lib/libtiling.so", run="op-package.run")

    def write(path, value):
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    host = {"x86_64": 62, "aarch64": 183}[platform.machine()]
    write(paths["api"], elf_fixture((recipe.host_entry, recipe.host_entry + "GetWorkspaceSize",
                                    "fi_aclnn_binding_fingerprint"), machine=host, kind=3))
    write("fi_aclnn_submit.so", elf_fixture(("fi_submit", "fi_submit_identity"), machine=host, kind=3))
    write_binding(directory, recipe)
    support = {"opMode": "dynamic"}
    for role in ("input", "output"):
        support[role + "s"] = [dict(name=p.name, dtype=p.dtype) for p in recipe.tensor_parameters if p.role == role]
    binary = elf_fixture((entry,), machine=0x1029, kind=2)
    write(paths["binary"], binary)
    write(paths["metadata"], dict(binFileSuffix=".o", core_type="AIC", coreType="AiCore",
        magic="RT_DEV_BINARY_MAGIC_ELF_AICUBE", sha256=hashlib.sha256(binary).hexdigest(),
        supportInfo=support, kernelList=[dict(kernelName=entry)]))
    write(paths["config"], {recipe.package_op_type: dict(binaryList=[dict(
        binPath=compute + "/probe/kernel.o", jsonPath=compute + "/probe/kernel.json")])})
    write(paths["tiling"], b"fixture auxiliary library, never loaded")
    write(paths["run"], b"fixture installer, never executed")
    write(spec.output_path, dict(schema_version=2, native_adapter="fi_aclnn_adapter.json",
        torch_abi=TEST_TORCH_ABI, output_adapter=recipe.output_adapter,
        spec_fingerprint=spec.fingerprint, recipe=recipe.to_dict(), vendor_path=vendor,
        api_library=paths["api"], compute_unit=compute, run_file=paths["run"]))
    return paths


@pytest.fixture
def generation(unit, tmp_path):
    recipe, _, spec = unit
    store = DiskJitBuildStore(tmp_path / "store")
    with store.lock(spec), store.staging(spec) as staging:
        paths = write_fixture(staging, recipe, spec)
        (staging / "build.log").write_text("synthetic verification fixture\n")
        record = store.publish(spec, staging, _verify_output(spec, staging))
    return store, record, paths


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []

    def cdll(path, mode):
        library = Path(path)
        calls.append(library)
        vendor = library.parents[2]
        assert any(p.is_dir() and p.samefile(vendor) for p in
                   map(Path, os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)))
        manifest = library.parents[5] / "op-package.json"
        recipe = BuildRecipe.from_dict(json.loads(manifest.read_text())["recipe"])
        return SimpleNamespace(fi_aclnn_binding_fingerprint=lambda: recipe.fingerprint.encode(),
                               **{recipe.host_entry: object(), recipe.host_entry + "GetWorkspaceSize": object()})

    monkeypatch.setattr("flashinfer.aclnn.build.torch_abi", lambda: TEST_TORCH_ABI)
    monkeypatch.setattr(C, "CDLL", cdll)
    monkeypatch.setattr(package, "_active_package", None)
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
    try:
        yield calls
    finally:
        for _, module in (package._active_package or {}).values():
            if module._registration_fd is not None:
                os.close(module._registration_fd)
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)


def test_all_output_variants_bind_their_complete_source_closure():
    recipes = load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
    for recipe in recipes:
        assert BuildRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe
        if recipe.implementation == "native":
            resolve_build_recipe(recipe, source_root=ROOT, toolchain_fingerprint="a" * 64)
    for implementation in ("native", "catlass"):
        selected = AscendCBuildConfig(implementation, target_soc="Ascend950").select(recipes, "matmul_probe")
        assert (selected.implementation, selected.output_kind) == (implementation, "op_package")


@pytest.mark.parametrize("soc,compute", [
    ("ascend910b1", "ascend910b"), ("ascend910_9391", "ascend910_93"),
    ("ascend310p1", "ascend310p"), ("ascend950pr_9579", "ascend950"),
])
def test_requested_soc_stays_bound_even_when_compute_unit_is_a_family(unit, soc, compute):
    recipe, _, _ = unit
    mapped = _for_target(recipe, soc)
    assert mapped.target_soc == soc and mapped.kernel_id.endswith("/" + soc)
    assert package.package_target(soc)[0] == compute
    assert mapped.fingerprint != recipe.fingerprint
    if not soc.startswith("ascend950"):
        recipes = load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
        assert AscendCBuildConfig(output_kind="op_package", target_soc=soc).select(recipes, "matmul_probe") == mapped


def test_unsupported_catlass_architecture_and_unknown_soc_fail_before_discovery(monkeypatch):
    recipes = load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
    catlass = AscendCBuildConfig("catlass", target_soc="Ascend950").select(recipes, "matmul_probe")
    with pytest.raises(SchemaError, match="Arch::Ascend950"):
        _for_target(catlass, "ascend310p1")
    with pytest.raises(SchemaError, match="unsupported.*SOC"):
        _for_target(catlass, "unrecognized")


@pytest.mark.parametrize("change", [dict(output_adapter="ascendc.host_shared.v1"),
    dict(host_sources=()), dict(package_vendor="../system"), dict(host_entry="wrong"), dict(tiling_mode="none")])
def test_package_protocol_cannot_be_reinterpreted_as_a_direct_launcher(unit, change):
    with pytest.raises(SchemaError):
        replace(unit[0], **change)


@pytest.mark.parametrize("kind,machine", [(1, 0x1029), (3, 62), (2, 62)])
def test_device_extension_never_substitutes_for_npu_et_exec(unit, tmp_path, kind, machine):
    recipe, _, spec = unit
    paths = write_fixture(tmp_path, recipe, spec)
    (tmp_path / paths["binary"]).write_bytes(elf_fixture((recipe.device_entry,), machine=machine, kind=kind))
    with pytest.raises(SchemaError, match="ELF type/machine mismatch"):
        _verify_output(spec, tmp_path)


def test_generated_metadata_symbols_must_exist_in_device_elf(unit, tmp_path):
    recipe, _, spec = unit
    paths = write_fixture(tmp_path, recipe, spec)
    (tmp_path / paths["binary"]).write_bytes(elf_fixture(("unrelated",), machine=0x1029, kind=2))
    with pytest.raises(SchemaError, match="metadata entries"):
        _verify_output(spec, tmp_path)


@pytest.mark.parametrize("change", ["../outside.o", "ascend910b/probe/kernel.o", "ascend950/../../outside.o"])
def test_binary_config_cannot_escape_or_switch_soc(unit, tmp_path, change):
    recipe, _, spec = unit
    paths = write_fixture(tmp_path, recipe, spec)
    config = json.loads((tmp_path / paths["config"]).read_text())
    config[recipe.package_op_type]["binaryList"][0]["binPath"] = change
    (tmp_path / paths["config"]).write_text(json.dumps(config))
    with pytest.raises(SchemaError):
        _verify_output(spec, tmp_path)


@pytest.mark.parametrize("corrupt", ["binary", "api", "config", "tiling", "metadata", "run", "extra"])
def test_every_release_file_is_verified_before_dlopen(generation, fake_runtime, tmp_path, corrupt):
    store, record, paths = generation
    release = export_op_package(store.directory(record), tmp_path / "release")
    target = release / paths[corrupt] if corrupt != "extra" else release / "unexpected-file"
    target.write_bytes(b"corrupted or injected")
    with pytest.raises(SchemaError, match="inventory mismatch"):
        package.load_op_package(release, target_soc=record.artifact.target_soc)
    assert fake_runtime == []


def test_export_and_load_verified_package_once_without_compiler_or_sources(generation, fake_runtime, tmp_path, monkeypatch):
    store, record, _ = generation
    release = export_op_package(store.directory(record), tmp_path / "release")
    assert not (release / "build.log").exists()
    assert verify_build_directory(release) == record

    def forbidden(*args, **kwargs):
        pytest.fail("AOT runtime must not discover tools, resolve sources, or compile")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(AscendCToolchain, "discover", forbidden)
    monkeypatch.setattr(AscendCBackend, "compile", forbidden)
    module = package.load_op_package(release, target_soc=record.artifact.target_soc)
    assert package.load_op_package(release, target_soc=record.artifact.target_soc) is module
    assert module.call_path == release / "fi_aclnn_submit.so"
    assert len(fake_runtime) == 1
    with pytest.raises(SchemaError, match="requested SOC"):
        package.load_op_package(release, target_soc="ascend910b1")


def test_aot_build_and_jit_cache_only_load_share_one_store(unit, tmp_path, fake_runtime, monkeypatch):
    recipe, backend, spec = unit
    store = DiskJitBuildStore(tmp_path)
    calls = []

    def fixture_compile(spec, *, build_directory):
        calls.append(spec.fingerprint)
        write_fixture(build_directory, recipe, spec)

    monkeypatch.setattr(backend, "compile", fixture_compile)
    record = _build_record(spec, backend=backend, store=store)
    loaded = build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex(),
                            policy=JitCompilationPolicy.CACHE_ONLY)
    assert calls == [spec.fingerprint]
    assert store.lookup(spec) == record
    assert record.artifact.format == ArtifactFormat.ASCENDC_OP_PACKAGE
    assert loaded.opaque_module.recipe == recipe


def test_different_package_cannot_silently_replace_process_registration(unit, generation, fake_runtime, tmp_path):
    store, record, _ = generation
    package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc)
    recipe, _, spec = unit
    other_spec = replace(spec, name="different", specialization_fingerprint="b" * 64)
    other_store = DiskJitBuildStore(tmp_path / "other")
    with other_store.lock(other_spec), other_store.staging(other_spec) as staging:
        write_fixture(staging, recipe, other_spec)
        other = other_store.publish(other_spec, staging, _verify_output(other_spec, staging))
    with pytest.raises(SchemaError, match="fresh process"):
        package.load_op_package(other_store.directory(other), target_soc=record.artifact.target_soc)
    assert len(fake_runtime) == 1
