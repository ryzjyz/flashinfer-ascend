"""Generated adaptation rejects a mismatched native ABI before compilation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "aclnn_adapter_generator", ROOT / "csrc/cmake/operator/aclnn_adapter.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture
def binding(tmp_path):
    from flashinfer.ascendc.gemm.matmul.implementation import IMPLEMENTATION
    recipe = IMPLEMENTATION.prepare_outputs(ROOT)["recipe"].to_dict()
    recipe["attributes"] = [dict(name="transpose_a", dtype="bool", default=False)]
    from flashinfer.jit.recipe import BuildRecipe
    recipe = BuildRecipe.from_dict(recipe).to_dict()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    header = tmp_path / "aclnn_fi_matmul.h"
    header.write_text("""#pragma once
#include <cstdint>
struct aclTensor;
struct aclOpExecutor;
using aclnnStatus = uint32_t;
using aclrtStream = void*;
extern "C" aclnnStatus aclnnFiMatmulGetWorkspaceSize(
  const aclTensor* a, const aclTensor* b, bool transposeA, const aclTensor* out,
  uint64_t* workspaceSize, aclOpExecutor** executor);
extern "C" aclnnStatus aclnnFiMatmul(void*, uint64_t, aclOpExecutor*, aclrtStream);
""")
    return recipe, path, header


def test_typed_adapter_binds_exact_operator_and_cann_header(binding, tmp_path):
    recipe, path, header = binding
    output = tmp_path / "generated.cpp"
    record = generator.generate(path, tmp_path, output)
    assert record["recipe_fingerprint"] == generator.fingerprint(recipe)
    assert record["header_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    assert record["source_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_suffix(".h").read_bytes() == header.read_bytes()
    call = output.with_name("fi_aclnn_submit.cpp").read_text()
    assert "EXEC_NPU_CMD(aclnnFiMatmul, p0, p1, a0, p2)" in call
    before = output.read_bytes()
    generator.generate(path, tmp_path, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("before,after", [
    ("aclTensor* a", "aclTensor* wrong"),
    ("bool transposeA", "double transposeA"),
    ("aclTensor* out", "aclTensor* result"),
    ("uint64_t* workspaceSize", "uint32_t* workspaceSize"),
])
def test_mismatched_opdef_is_rejected_without_publishing_adapter(binding, tmp_path, before, after):
    _, path, header = binding
    header.write_text(header.read_text().replace(before, after))
    output = tmp_path / "rejected.cpp"
    with pytest.raises(ValueError, match="does not match NativeBinding"):
        generator.generate(path, tmp_path, output)
    assert not output.exists()


def test_missing_or_duplicate_generated_entry_fails_closed(binding, tmp_path):
    _, path, header = binding
    (tmp_path / "aclnn_duplicate.h").write_bytes(header.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        generator.generate(path, tmp_path, tmp_path / "rejected.cpp")


def test_optional_input_and_multiple_outputs_use_exact_cann_aliases(binding, tmp_path):
    recipe, path, header = binding
    recipe["tensor_parameters"][1]["optional"] = True
    recipe["tensor_parameters"].append(dict(recipe["tensor_parameters"][-1], name="scale"))
    path.write_text(json.dumps(recipe))
    text = header.read_text().replace("aclTensor* b,", "aclTensor* bOptional,")
    text = text.replace("aclTensor* out,", "aclTensor* outOut, const aclTensor* scaleOut,")
    header.write_text(text)
    generator.generate(path, tmp_path, tmp_path / "accepted.cpp")
    header.write_text(text.replace("aclTensor* outOut, const aclTensor* scaleOut",
                                   "aclTensor* scaleOut, const aclTensor* outOut"))
    with pytest.raises(ValueError, match="does not match NativeBinding"):
        generator.generate(path, tmp_path, tmp_path / "reordered.cpp")


def test_optional_string_alias_preserves_attribute_type_and_order(binding, tmp_path):
    recipe, path, header = binding
    recipe["attributes"] = [*recipe["attributes"], dict(name="layout", dtype="string", default="ND")]
    path.write_text(json.dumps(recipe))
    text = header.read_text().replace("bool transposeA,", "bool transposeA, const char* layoutOptional,")
    header.write_text(text)
    generator.generate(path, tmp_path, tmp_path / "accepted.cpp")
    header.write_text(text.replace("char* layoutOptional", "aclTensor* layoutOptional"))
    with pytest.raises(ValueError, match="does not match NativeBinding"):
        generator.generate(path, tmp_path, tmp_path / "wrong_type.cpp")


def test_packaged_adapter_roundtrips_native_shapes_and_rejects_another_recipe(binding, tmp_path):
    from dataclasses import replace
    from flashinfer.jit.op_package import _verify_native_adapter
    from flashinfer.jit.recipe import BuildRecipe
    from flashinfer.runtime import SchemaError

    raw, path, _ = binding
    recipe = BuildRecipe.from_dict(raw)
    generator.generate(path, tmp_path, tmp_path / "fi_aclnn_adapter.cpp")
    data = dict(native_adapter="fi_aclnn_adapter.json")
    symbols = {"fi_aclnn_binding_fingerprint"}
    _verify_native_adapter(tmp_path, data, recipe, symbols)
    with pytest.raises(SchemaError, match="does not match"):
        _verify_native_adapter(tmp_path, data, replace(recipe, kernel_id=recipe.kernel_id + "_other"), symbols)


def test_dynamic_slots_preserve_cann_members_and_declared_dtype_combinations(binding):
    from dataclasses import replace
    from flashinfer.jit.op_package import _binary_tensor_signatures
    from flashinfer.jit.recipe import BuildRecipe, ShapeConstraint, TensorParameter
    from flashinfer.runtime import SchemaError

    recipe = replace(BuildRecipe.from_dict(binding[0]), attributes=(),
                     tensor_parameters=(TensorParameter("xs", "input", "float32", ("N",), tensor_list=True),
                                        TensorParameter("out", "output", "float32", ("M",))),
                     shape_constraints=(ShapeConstraint("N", 1, 8), ShapeConstraint("M", 1, 16)),
                     dtype_signatures=(("float32", "float32"), ("float16", "float32")))
    dtype = {"xs": ("float32", "float16"), "out": "float32"}
    assert recipe.validate_tensor_metadata(shapes={"xs": ((2,), (3,)), "out": (5,)}, dtypes=dtype) == {
        "N": (2, 3), "M": 5}
    with pytest.raises(SchemaError, match="outside declared range"):
        recipe.validate_tensor_metadata(shapes={"xs": ((2,), (9,)), "out": (11,)}, dtypes=dtype)
    with pytest.raises(SchemaError, match="not a declared combination"):
        recipe.validate_tensor_metadata(shapes={"xs": ((2,), (3,)), "out": (5,)},
                                        dtypes={"xs": ("int8", "float16"), "out": "float32"})
    members = [dict(name="xs", paramType="dynamic", format="ND", dtype=t) for t in dtype["xs"]]
    output = dict(name="out", paramType="required", format="ND", dtype="float32")
    info = dict(inputs=[members], outputs=[output])
    assert set(_binary_tensor_signatures(recipe, info)) == set(recipe.native_dtype_signatures)
    members[1]["paramType"] = "required"
    with pytest.raises(SchemaError, match="format/optionality"):
        _binary_tensor_signatures(recipe, info)


@pytest.mark.parametrize("kind,value,invalid", [
    ("int8", -128, -129), ("uint16", 65535, 65536), ("uint64", (1 << 64) - 1, -1),
    ("float32", 0.25, 1e39), ("bool_array", [False, True], [0, 1]),
    ("float_array", (1, 2.5), ("1",)), ("int_array", [-2, 0], [False]), ("dtype", "int4", "unknown"),
])
def test_attribute_schema_roundtrip_and_native_range(kind, value, invalid):
    from flashinfer.jit.aclnn_contract import AttributeParameter
    from flashinfer.runtime import SchemaError
    param = AttributeParameter("v", kind, value)
    assert AttributeParameter(**json.loads(json.dumps(param.to_dict()))) == param
    with pytest.raises(SchemaError):
        param.validate(invalid)
