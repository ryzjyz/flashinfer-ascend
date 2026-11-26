"""Standard CANN op-package verification and explicit compiler-free ACLNN load.

No SDK discovery, optional imports, installers or filesystem work at import.
The release directory is a copy of a verified JitBuildStore generation, not a
second cache. Each vendor/op type retains one immutable identity per process;
distinct operators can coexist, including lazy activation after NPU init.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
from threading import RLock
from typing import Optional

from flashinfer.runtime import ArtifactFormat, SchemaError

from ._elf import elf_symbols
from .core import JitSpec, _canonical_hash
from .recipe import _Recipe, _matching_signatures, _recipe_from_artifact, _path


OP_PACKAGE_ADAPTER_HASH = _canonical_hash("ascendc.op_package.v1")


class OpPackageSymbolError(SchemaError):
    """An otherwise verified package cannot supply its required ACLNN entry."""


def package_target(soc):
    """Pure requested-product -> CANN compute-unit/architecture mapping.

    A family mapping does not assert successful compilation or device support.
    The full requested product is still bound into the recipe/spec/release.
    """
    key = soc.lower()
    from flashinfer.runtime.targets import canonical_soc_model, SOC_TARGETS

    if key == "ascend310p1":
        return "ascend310p", "dav-2002"
    try:
        return SOC_TARGETS[canonical_soc_model(soc)]
    except ValueError as error:
        raise SchemaError("unsupported op-package SOC: " + soc) from error


def _file(root, relative):
    _path(relative)
    root = Path(root).resolve(strict=True)
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise SchemaError("op-package path escapes its immutable root: " + relative)
    return path


def collect_package(spec, recipe, build, directory, *, source_root, toolchain):
    """Collect only CPack's staged standard vendor tree and installer, never run it."""
    vendors = list(build.glob("_CPack_Packages/*/External/*.run/packages/vendors/" + recipe.package_vendor))
    installers = list(build.glob("*.run"))
    if len(vendors) != 1 or len(installers) != 1:
        raise SchemaError("CANN must produce one staged standard vendor tree and one .run")
    shutil.copytree(vendors[0].parent.parent, directory / "packages")
    shutil.copy2(installers[0], directory / "op-package.run")
    vendor = "packages/vendors/" + recipe.package_vendor
    data = dict(schema_version=1, output_adapter=recipe.output_adapter,
        spec_fingerprint=spec.fingerprint, recipe=recipe.to_dict(),
        vendor_path=vendor, api_library=vendor + "/op_api/lib/libcust_opapi.so",
        compute_unit=package_target(recipe.target_soc)[0], run_file="op-package.run")
    if recipe.schema_version == 2:
        for suffix in (".cpp", ".json", ".h"):
            shutil.copyfile(build / ("fi_aclnn_adapter" + suffix),
                            directory / ("fi_aclnn_adapter" + suffix))
        from flashinfer.aclnn.build import build_package_call, torch_abi
        build_package_call(build / "fi_aclnn_submit.cpp", directory / "fi_aclnn_submit.so", source_root,
                           cann_root=toolchain.cann_root, compiler=toolchain.host_compiler)
        data.update(schema_version=2, native_adapter="fi_aclnn_adapter.json", torch_abi=torch_abi())
    (directory / spec.output_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    verify_op_package(directory, spec)


def verify_op_package(directory, spec):
    """Verify manifest binding, ACLNN host ELF, config references and NPU ET_EXEC.

    A direct device -c ET_REL and a host ET_DYN are both rejected as package
    kernels. Extensions and CANN's filename hash never select the loader.
    The surrounding store/release manifest verifies every auxiliary byte.
    """
    if (spec.generator_id != "ascendc" or spec.output_path != "op-package.json"
            or dict(spec.input_fingerprints).get("output_adapter") != OP_PACKAGE_ADAPTER_HASH):
        raise SchemaError("spec does not declare the op-package adapter")
    data = json.loads(_file(directory, spec.output_path).read_text())
    fields = {"schema_version", "output_adapter", "spec_fingerprint", "recipe",
              "vendor_path", "api_library", "compute_unit", "run_file"}
    if data.get("schema_version") == 2:
        fields.update(("native_adapter", "torch_abi"))
    if set(data) != fields:
        raise SchemaError("invalid op-package manifest fields")
    recipe = _recipe_from_artifact(data["recipe"])
    vendor = "packages/vendors/" + str(recipe.package_vendor)
    compute = package_target(recipe.target_soc)[0]
    if (data["schema_version"] not in (1, 2) or data["output_adapter"] != "ascendc.op_package.v1"
            or recipe.output_kind != "op_package" or data["spec_fingerprint"] != spec.fingerprint
            or recipe.target_soc != spec.target_soc or recipe.source != spec.source
            or recipe.kernel_id != spec.kernel_id or spec.entry_points != (recipe.host_entry,)
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint
            or data["compute_unit"] != compute or data["vendor_path"] != vendor
            or data["api_library"] != vendor + "/op_api/lib/libcust_opapi.so"
            or data["run_file"] != "op-package.run"):
        raise SchemaError("op-package manifest does not bind recipe/spec/layout")
    machines = {"x86_64": 62, "aarch64": 183}
    symbols = elf_symbols(_file(directory, data["api_library"]).read_bytes(),
        machine=machines[platform.machine()], elf_type=3, dynamic_only=True)
    for entry in (recipe.host_entry, recipe.host_entry + "GetWorkspaceSize"):
        if entry not in symbols:
            raise OpPackageSymbolError("op-package host ELF has no ACLNN symbol: " + entry)
    if data["schema_version"] == 2:
        _verify_native_adapter(directory, data, recipe, symbols)
        submit = elf_symbols(_file(directory, "fi_aclnn_submit.so").read_bytes(),
                             machine=machines[platform.machine()], elf_type=3, dynamic_only=True)
        if not {"fi_submit", "fi_submit_identity"} <= submit:
            raise OpPackageSymbolError("Missing typed ACLNN submission entry")
    kernel = vendor + "/op_impl/ai_core/tbe/kernel/"
    config = json.loads(_file(directory, kernel + "config/" + compute + "/binary_info_config.json").read_text())
    if set(config) != {recipe.package_op_type}:
        raise SchemaError("binary_info_config must identify exactly the declared op type")
    binaries = config[recipe.package_op_type]["binaryList"]
    if not binaries:
        raise SchemaError("op-package contains no precompiled kernel")
    dtype_signatures = set()
    for item in binaries:
        for key in ("binPath", "jsonPath"):
            if not item[key].startswith(compute + "/"):
                raise SchemaError("binary_info_config references a different compute unit")
        binary = _file(directory, kernel + item["binPath"]).read_bytes()
        entries = elf_symbols(binary, machine=0x1029, elf_type=2)
        metadata = json.loads(_file(directory, kernel + item["jsonPath"]).read_text())
        # CANN renames the source entry to <OpType>_<digest>_<tiling-key>.
        # Check the actual names recorded in its binary metadata against ELF,
        # instead of treating the source C symbol as the packaged entry.
        generated = [k["kernelName"] for k in metadata.get("kernelList", ())]
        mixed = recipe.schema_version == 2 and recipe.task_type == "aic_aiv"
        vector_sync = recipe.schema_version == 2 and recipe.task_type == "aiv_hard_sync"
        suffixes = ("_mix_aic", "_mix_aiv") if mixed else ("_mix_aiv",) if vector_sync else ("",)
        required_entries = [name + suffix for name in generated for suffix in suffixes]
        if (not generated or any(not name.startswith(recipe.package_op_type + "_") for name in generated)
                or any(name not in entries for name in required_entries)):
            raise SchemaError("op-package NPU ET_EXEC does not define its CANN metadata entries")
        if mixed:
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "MIX"
                    or metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF"
                    or metadata.get("intercoreSync") != 1
                    or any(k.get("kernelType") not in {"MIX_AIC", "MIX_AIV"}
                           or k.get("crossCoreSync") != 1
                           for k in metadata["kernelList"])):
                raise SchemaError("op-package kernel metadata does not declare the v2 synchronized AIC+AIV executable contract")
        elif vector_sync:
            # CANN encodes synchronized AIV execution as a MIX launch with zero AICs.
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "MIX"
                    or metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF" or metadata.get("intercoreSync") != 1
                    or any(k.get("kernelType") not in {"MIX_AIC", "MIX_AIV"} or k.get("taskRation") != "0:1"
                           or k.get("crossCoreSync") != 1 for k in metadata["kernelList"])):
                raise SchemaError("op-package kernel metadata does not declare synchronized AIV execution")
        elif recipe.schema_version == 2 and recipe.task_type == "aiv_only":
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "VectorCore"
                    or metadata.get("core_type") != "AIV" or metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF_AIVEC"):
                raise SchemaError("op-package kernel metadata does not declare the AIV executable contract")
        else:
            # v1 retains its AIC-only metadata rules unchanged. 310P has unified
            # AiCore metadata without the A2/A3/A5 core_type field.
            magic = "RT_DEV_BINARY_MAGIC_ELF" if compute == "ascend310p" else "RT_DEV_BINARY_MAGIC_ELF_AICUBE"
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "AiCore"
                    or metadata.get("magic") != magic
                    or (compute != "ascend310p" and metadata.get("core_type") != "AIC")):
                raise SchemaError("op-package kernel metadata does not declare the target's AIC executable contract")
        if recipe.schema_version == 2:
            support = metadata.get("supportInfo", {})
            if support.get("opMode") != "dynamic":
                raise SchemaError("v2 package metadata must declare dynamic host tiling")
            dtype_signatures.update(_binary_tensor_signatures(recipe, support))
            if recipe.extended_aclnn:
                attributes = support.get("attrs", ())
                expected = [(a.name, a.opdef_dtype)
                            for a in recipe.attributes]
                if [(a.get("name"), a.get("dtype")) for a in attributes] != expected:
                    raise SchemaError("extended package attribute order/types differ from recipe")
            if (recipe.workspace_bytes == "dynamic"
                    and metadata.get("workspace", {}).get("size") != [-1]):
                raise SchemaError("v2 dynamic workspace must have a host-computed metadata size")
        if metadata.get("sha256") != hashlib.sha256(binary).hexdigest():
            raise SchemaError("CANN kernel metadata digest does not match its NPU executable")
    if recipe.schema_version == 2 and dtype_signatures != set(recipe.native_dtype_signatures):
        raise SchemaError("v2 package is missing a declared dtype signature executable")
    _file(directory, data["run_file"])
    return data, recipe


def _binary_tensor_signatures(recipe, support):
    """CANN represents each dynamic OpDef slot as a group of tensor records."""
    choices = []
    for role, key in (("input", "inputs"), ("output", "outputs")):
        declared = [p for p in recipe.tensor_parameters if p.role == role]
        slots = support.get(key, ())
        if len(slots) != len(declared):
            raise SchemaError("v2 package tensor parameter order/types differ from recipe: " + role)
        for parameter, slot in zip(declared, slots):
            members = slot if isinstance(slot, list) else [slot]
            if (not members or not parameter.tensor_list and isinstance(slot, list)
                    or any(not isinstance(p, dict) or p.get("name") != parameter.name for p in members)):
                raise SchemaError("v2 package tensor parameter order/types differ from recipe: " + role)
            if recipe.extended_aclnn:
                expected = "dynamic" if parameter.tensor_list else "optional" if parameter.optional else "required"
                for p in members:
                    layout = {"ND": "ND", "FRACTAL_NZ": "NZ"}.get(p.get("format"))
                    if layout not in parameter.physical_layouts or p.get("paramType") != expected:
                        raise SchemaError("extended package Tensor format/optionality differs from recipe")
            choices.append({p.get("dtype") for p in members})
    return _matching_signatures(choices, recipe.native_dtype_signatures,
                                "v2 package tensor parameter order/types differ from recipe: dtype signature")


def _verify_native_adapter(directory, data, recipe, symbols):
    """Match generated adaptation to this exact native ABI before loading code."""
    if recipe.schema_version != 2 or data["native_adapter"] != "fi_aclnn_adapter.json":
        raise SchemaError("typed ACLNN adapter requires its exact v2 recipe")
    record = json.loads(_file(directory, data["native_adapter"]).read_text())
    fields = {"schema_version", "kernel_id", "recipe_fingerprint", "host_entry",
              "tensor_parameters", "attributes", "header_sha256", "source_sha256"}
    expected = json.loads(json.dumps(recipe.to_dict()))
    if (set(record) != fields or record["schema_version"] != 1
            or record["kernel_id"] != recipe.kernel_id
            or record["recipe_fingerprint"] != recipe.fingerprint
            or record["host_entry"] != recipe.host_entry
            or record["tensor_parameters"] != expected["tensor_parameters"]
            or record["attributes"] != expected.get("attributes", [])):
        raise SchemaError("generated ACLNN adapter does not match the operator NativeBinding")
    for suffix, key in ((".h", "header_sha256"), (".cpp", "source_sha256")):
        payload = _file(directory, "fi_aclnn_adapter" + suffix).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record[key]:
            raise SchemaError("generated ACLNN adapter source/header digest mismatch")
    if "fi_aclnn_binding_fingerprint" not in symbols:
        raise OpPackageSymbolError("op-package is missing its generated ACLNN adapter symbols")


_load_lock = RLock()
_active_package = None


@dataclass(frozen=True)
class OpPackageModule:
    """Owns a verified vendor library and its typed Torch submission library."""

    recipe: _Recipe
    vendor_path: Path
    api: object
    # Retain the directory descriptor backing CANN's dot-free /proc alias for
    # the active module's process lifetime. No second copy/cache is created.
    registration_path: Optional[Path] = None
    _registration_fd: Optional[int] = field(default=None, repr=False, compare=False)
    call_path: Optional[Path] = None



def load_op_packages(directories, *, target_soc):
    """Activate a complete, verified CANN vendor set before its first execution.

    CANN snapshots custom vendor paths. New vendors cannot be appended to an
    active set; prepare them together or use a fresh process. Each op/API and
    vendor has one immutable generation. No installer or compiler is invoked.
    Current CANN runtime version comparison remains an independent known gap.
    """
    import ctypes as C
    from .cache import verify_build_directory

    directories = tuple(dict.fromkeys(Path(path).resolve(strict=True) for path in directories))
    if not directories:
        raise SchemaError("op-package activation requires at least one directory")
    verified = []
    for directory in directories:
        record = verify_build_directory(directory)
        if (record.artifact.format != ArtifactFormat.ASCENDC_OP_PACKAGE
                or record.artifact.target_soc != target_soc):
            raise SchemaError("release is not an op-package for the requested SOC")
        spec = JitSpec.from_dict(json.loads((directory / "manifest.json").read_text())["spec"])
        data, recipe = verify_op_package(directory, spec)
        verified.append((directory, record, data, recipe))

    global _active_package
    with _load_lock:
        registrations = _active_package if _active_package is not None else {}
        identities = {name: (identity, module.recipe) for name, (identity, module) in registrations.items()}
        pending = {}
        for directory, record, data, recipe in verified:
            prior = identities.get(recipe.package_vendor)
            if prior is not None:
                if prior[0] != record.fingerprint:
                    raise SchemaError("CANN vendor registration is process-scoped; load another generation in a fresh process")
                continue
            for _, loaded_recipe in identities.values():
                if (loaded_recipe.package_op_type == recipe.package_op_type
                        or loaded_recipe.host_entry == recipe.host_entry):
                    raise SchemaError("CANN op type/ACLNN entry is already registered; use a fresh process")
            identities[recipe.package_vendor] = record.fingerprint, recipe
            pending[recipe.package_vendor] = directory, record, data, recipe
        if registrations and pending:
            raise SchemaError("CANN vendor paths are already active; prepare all operators together in a fresh process")
        previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        aliases = {}
        loaded = {}
        try:
            # Set every verified vendor path before the first host dlopen/API;
            # appending paths after the first custom execution is too late.
            for name, (directory, record, data, recipe) in pending.items():
                vendor = _file(directory, data["vendor_path"])
                registration_path, registration_fd = vendor, None
                if "." in str(vendor):
                    # CANN 9.2 derives binary JSON paths from the first dot.
                    # Retain a dot-free /proc alias to the SAME verified bytes.
                    registration_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    registration_path = (Path("/proc") / str(os.getpid()) / "fd"
                                         / str(registration_fd) / data["vendor_path"])
                aliases[name] = vendor, registration_path, registration_fd
                if not registration_path.is_dir():
                    raise SchemaError("CANN op-package dotted paths require Linux /proc directory fd access")
            if pending:
                paths = [str(alias[1]) for alias in aliases.values()]
                if previous:
                    paths.append(previous)
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(paths)
            for name, (directory, record, data, recipe) in pending.items():
                if data["schema_version"] == 2:
                    from flashinfer.aclnn.build import torch_abi
                    if data["torch_abi"] != torch_abi():
                        raise SchemaError("typed ACLNN package Torch/torch_npu build identity mismatch")
                api = C.CDLL(str(_file(directory, data["api_library"])), mode=C.RTLD_LOCAL)
                if data["schema_version"] == 2:
                    identity = api.fi_aclnn_binding_fingerprint
                    identity.argtypes, identity.restype = [], C.c_char_p
                    if identity() != recipe.fingerprint.encode("ascii"):
                        raise SchemaError("loaded ACLNN adapter identity differs from its operator")
                # Resolve both vendor entries before committing this registration or its fd.
                # ctypes caches the two verified function objects on the library handle.
                getattr(api, recipe.host_entry + "GetWorkspaceSize")
                getattr(api, recipe.host_entry)
                vendor, registration_path, registration_fd = aliases[name]
                loaded[name] = record.fingerprint, OpPackageModule(
                    recipe, vendor, api, registration_path, registration_fd,
                    _file(directory, "fi_aclnn_submit.so") if recipe.schema_version == 2 else None)
        except Exception:
            for _, _, fd in aliases.values():
                if fd is not None:
                    os.close(fd)
            if previous is None:
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
            else:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = previous
            raise
        if loaded:
            registrations = dict(registrations, **loaded)
            _active_package = registrations
        return tuple(registrations[recipe.package_vendor][1] for _, _, _, recipe in verified)


def load_op_package(directory, *, target_soc):
    """Load one package; callers needing several prepare one activation group."""
    return load_op_packages((directory,), target_soc=target_soc)[0]


__all__ = ["OpPackageModule", "OpPackageSymbolError", "load_op_package", "load_op_packages"]
