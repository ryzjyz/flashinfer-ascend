"""Generate a typed adapter from one recipe and its CANN-generated ACLNN header.

This build input uses only the standard library and runs after OpDef codegen.
No hand-authored operator switch or SDK probing is involved.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def _name(value):
    return value.replace("_", "").lower()


_NATIVE_INTEGERS = {f"{prefix}int{bits}": f"{prefix}int{bits}_t"
                    for prefix in ("", "u") for bits in (8, 16, 32, 64)}


def workspace_signature(recipe, headers):
    entry = recipe["host_entry"] + "GetWorkspaceSize"
    matches = []
    for path in sorted(Path(headers).glob("aclnn_*.h")):
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        match = re.search(r"\b" + re.escape(entry) + r"\s*\((.*?)\)\s*;", text, re.S)
        if match:
            matches.append((path, match[1]))
    if len(matches) != 1:
        raise ValueError("expected exactly one generated ACLNN header for " + entry)
    header, signature = matches[0]
    arguments = []
    for argument in signature.split(","):
        argument = " ".join(argument.split())
        match = re.fullmatch(r"(.+?[ *])([A-Za-z_][A-Za-z0-9_]*)", argument)
        if match is None:
            raise ValueError("unsupported generated ACLNN argument: " + argument)
        arguments.append((match[1].strip(), match[2]))
    tensors = recipe["tensor_parameters"]
    # CANN's C API appends Optional to optional input names and Out to each
    # output name when there are multiple outputs. Bind these exact aliases,
    # without accepting arbitrary names or reordering identical tensor types.
    expected = [(p["name"] + ("Optional" if p.get("optional", False) else ""), "aclTensorList*" if p.get("tensor_list") else "aclTensor*")
                for p in tensors if p["role"] == "input"]
    types = {**_NATIVE_INTEGERS, "bool": "bool", "float32": "float", "float64": "double", "string": "char*",
             "int_array": "aclIntArray*", "float_array": "aclFloatArray*", "bool_array": "aclBoolArray*",
             "scalar": "aclScalar*", "dtype": "aclDataType"}
    expected += [(p["name"], types[p["dtype"]]) for p in recipe.get("attributes", ())]
    outputs = [p for p in tensors if p["role"] == "output"]
    expected += [(p["name"] + ("Out" if len(outputs) > 1 else ""), "aclTensorList*" if p.get("tensor_list") else "aclTensor*") for p in outputs]
    expected += [("workspaceSize", "uint64_t*"), ("executor", "aclOpExecutor**")]
    actual = [(_name(name), re.sub(r"\s|\bconst\b", "", dtype)) for dtype, name in arguments]
    spellings = {"int": "int32_t", "unsignedint": "uint32_t", "short": "int16_t",
                 "unsignedshort": "uint16_t", "signedchar": "int8_t", "unsignedchar": "uint8_t"}
    actual = [(name, spellings.get(dtype, dtype)) for name, dtype in actual]
    wanted = [(_name(name), dtype) for name, dtype in expected]
    # CANN canonicalizes a single output's C parameter name to "out".
    # Its position and type remain exact; multiple outputs are never reordered.
    if len(outputs) == 1 and len(actual) == len(wanted) and actual[-3][0] == "out":
        actual[-3] = (wanted[-3][0], actual[-3][1])
    # Optional string attributes are nullable pointers in CANN's C signature;
    # scalar optional attributes keep their ordinary parameter names.
    inputs = sum(p["role"] == "input" for p in tensors)
    for index, attribute in enumerate(recipe.get("attributes", ()), inputs):
        if ((attribute["dtype"] == "string" or attribute.get("optional", False)) and index < len(actual)
                and actual[index][0] == wanted[index][0] + "optional"):
            actual[index] = (wanted[index][0], actual[index][1])
    if actual != wanted:
        raise ValueError("generated ACLNN ABI does not match NativeBinding: expected %r, got %r"
                         % (wanted, actual))
    return header, arguments


def validate_device_entry(recipe, ops_info):
    """Bind the author's entry name to OpDef codegen without rewriting C++."""
    info = configparser.ConfigParser(interpolation=None)
    info.optionxform = str
    with Path(ops_info).open() as source:
        info.read_file(source)
    op_type = recipe["package_op_type"]
    actual = info.get(op_type, "opInterface.value", fallback=None)
    if actual != recipe["device_entry"]:
        raise ValueError("OpDef device entry does not match BuildConfig.device_entry: "
                         "op_type=%s, declared=%s, generated=%s"
                         % (op_type, recipe["device_entry"], actual))


def torch_call(recipe):
    """Generate the same C++ EXEC_NPU_CMD call an operator author would write."""
    lines = [
        '#include "aclnn_adapter.h"',
        'using namespace flashinfer_aclnn;',
        '#define FI_EXPORT extern "C" __attribute__((visibility("default")))',
        'FI_EXPORT const char* fi_submit_identity() { return "' + fingerprint(recipe) + '"; }',
        'FI_EXPORT void fi_submit(const Call& invocation) {',
        '    Operands operands(invocation);',
        '    TORCH_CHECK(invocation.attributes.size() == %d, "ACLNN attribute count mismatch");'
        % len(recipe.get("attributes", ())),
    ]
    params = recipe["tensor_parameters"]
    slots = {param["name"]: index for index, param in enumerate(params)}
    arguments = []
    def tensor(i, param):
        listed, optional = param.get("tensor_list", False), param.get("optional", False)
        output = param["role"] == "output"
        flags = [output, listed, optional]
        fmt = sum({"ND": 1, "NZ": 2}[f] for f in param.get("physical_layouts", ("ND",)))
        flags = ", ".join(str(b).lower() for b in flags)
        cpu = str(param.get("device") == "cpu").lower()
        strided = ", true" if param.get("allow_strided") else ""
        lines.append(f'    auto t{i} = operands.take({flags}, {fmt}, {cpu}{strided});')
        if listed:
            value = f't{i}' if optional else f'*t{i}'
        elif optional:
            value = f't{i} ? t{i}->front() : at::Tensor()'
        else:
            value = f't{i}->front()'
        conversion = param.get("native_conversions", ())
        if conversion:
            kind = {"int4": "Int4", "uint64": "UInt64", "float4_e2m1": "Float4E2M1",
                    "float8_e8m0": "Float8E8M0"}[conversion[0]["native_dtype"]]
            value = f'StorageView{{{value}, StorageKind::{kind}}}'
        lines.append(f'    auto p{i} = {value};')
        arguments.append(f'p{i}')
    for i, param in enumerate(params):
        if param["role"] == "input":
            tensor(i, param)
    for i, attr in enumerate(recipe.get("attributes", ())):
        dtype, optional = attr["dtype"], attr.get("optional", False)
        value = f'v{i}'
        lines.append(f'    const auto {value} = invocation.attributes.get({i});')
        if dtype in ("int_array", "float_array", "bool_array"):
            cpp = {"int_array":"int64_t", "float_array":"float", "bool_array":"bool"}[dtype]
            ctor = f'BoolValues({value})' if dtype == "bool_array" else f'array_values<{cpp}>({value})'
            if optional:
                lines.append(f'    auto data{i} = {value}.isNone() ? std::vector<{cpp}>() : {ctor};')
                expr = f'{value}.isNone() ? c10::nullopt : c10::optional<at::ArrayRef<{cpp}>>(data{i})'
            else:
                lines.append(f'    auto data{i} = {ctor};')
                expr = f'data{i}.view()' if dtype == "bool_array" else f'at::ArrayRef<{cpp}>(data{i})'
        elif dtype in _NATIVE_INTEGERS:
            expr = f'static_cast<{_NATIVE_INTEGERS[dtype]}>({value}.toInt())'
        else:
            expr = {"bool":f'{value}.toBool()', "float32":f'static_cast<float>({value}.toDouble())', "float64":f'{value}.toDouble()',
                    "string":f'{value}.toStringRef().c_str()', "scalar":f'{value}.toScalar()',
                    "dtype":f'{value}.toScalarType()'}[dtype]
            if optional:
                expr = f'{value}.isNone() ? c10::nullopt : c10::optional<at::Scalar>({expr})'
        lines.append(f'    auto a{i} = {expr};')
        arguments.append(f'a{i}')
    for i, param in enumerate(params):
        if param["role"] == "output":
            tensor(i, param)
    # Use the consumed Tensor slices so optional and TensorList widths are resolved.
    for i, param in enumerate(params):
        if param.get("alias_of") is not None:
            lines.append(f'    operands.exact_alias(*t{slots[param["alias_of"]]}, *t{i});')
    lines += ['    operands.finish();', '    EXEC_NPU_CMD(' + recipe["host_entry"] + ', ' + ', '.join(arguments) + ');', '}']
    return "\n".join(lines) + "\n"


def generate(recipe_file, headers, output, *, ops_info=None):
    recipe = json.loads(Path(recipe_file).read_text())
    if recipe.get("schema_version") != 2:
        raise ValueError("typed ACLNN generation requires a v2 recipe")
    if ops_info is not None:
        validate_device_entry(recipe, ops_info)
    header, _ = workspace_signature(recipe, headers)
    digest = fingerprint(recipe)
    entry = recipe["host_entry"]
    source = (
        '// Generated from NativeBinding and the matching CANN OpDef header.\n'
        '#include "' + header.name + '"\n'
        '#define FI_EXPORT extern "C" __attribute__((visibility("default")))\n'
        'FI_EXPORT const char* fi_aclnn_binding_fingerprint() { return "' + digest + '"; }\n'
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    output.with_name("fi_aclnn_submit.cpp").write_text(torch_call(recipe))
    metadata = dict(schema_version=1, kernel_id=recipe["kernel_id"],
                    recipe_fingerprint=digest, host_entry=entry,
                    tensor_parameters=recipe["tensor_parameters"],
                    attributes=recipe.get("attributes", []),
                    header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    output.with_suffix(".h").write_bytes(header.read_bytes())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--headers", required=True)
    parser.add_argument("--ops-info", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.recipe, args.headers, args.output, ops_info=args.ops_info)
