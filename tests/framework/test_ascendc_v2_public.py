"""Only explicit v2 ACLNN contracts can be authored, selected or published."""

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer.aot.build import export_op_package
from flashinfer._resources import data_root, recipe_root
from flashinfer.jit import AscendCBuildConfig, BuildRecipe, load_build_recipes, resolve_build_recipe
from flashinfer.jit.core import JitSpec
from flashinfer.jit.op_package import OP_PACKAGE_ADAPTER_HASH, OpPackageModule
from flashinfer.jit.torch_library import matmul_probe_op
from flashinfer.runtime import SchemaError
from tests.packaging.test_ascendc_op_package import generation, unit

from tests.production_inventory import PUBLIC_RECIPES, PRODUCTION_OPS

ROOT = Path(__file__).resolve().parents[2]


def test_public_manifest_set_is_exactly_v2_and_preserves_production_inventory():
    recipes = tuple(r for p in sorted({p for folder in (recipe_root(ROOT), ROOT / 'csrc/recipes') for p in folder.glob('*.json')}) for r in load_build_recipes(p))
    assert {(r.recipe_id, r.implementation, r.target_soc, r.schema_version, r.output_kind) for r in recipes} == PUBLIC_RECIPES
    records = json.loads((data_root(ROOT) / 'kernels/registry.json').read_text())['kernels']
    assert {r['op'] for r in records if r['backend'] == 'ascendc'} == PRODUCTION_OPS
    assert inspect.signature(BuildRecipe).parameters['schema_version'].default == 2
    assert AscendCBuildConfig().output_kind == 'op_package'


def test_public_construction_manifest_config_and_release_reject_v1(generation, unit, tmp_path, monkeypatch):
    recipe = unit[0]
    old = dict(recipe.to_dict(), schema_version=1)
    with pytest.raises(SchemaError, match="schema_version=2"):
        BuildRecipe.from_dict(old)
    manifest = tmp_path / "old.json"
    manifest.write_text(json.dumps([old]))
    with pytest.raises(SchemaError, match="schema_version=2"):
        load_build_recipes(manifest)
    for output in ("host_shared", "host_shared.v1"):
        with pytest.raises(SchemaError):
            AscendCBuildConfig(output_kind=output)
    # Exercise the public rejection boundaries without building an obsolete operator.
    from types import SimpleNamespace
    old_recipe = SimpleNamespace(**old)
    with pytest.raises(SchemaError, match="legacy v1"):
        AscendCBuildConfig().select((old_recipe,), "matmul_probe")
    with pytest.raises(SchemaError, match="legacy artifacts are read-only"):
        resolve_build_recipe(old_recipe, source_root=ROOT, toolchain_fingerprint="a" * 64)
    store, record, _ = generation
    monkeypatch.setattr("flashinfer.aot.build.verify_op_package", lambda *args: ({}, old_recipe))
    with pytest.raises(SchemaError, match="new releases require a v2 recipe"):
        export_op_package(store.directory(record), tmp_path / "public-release")
    assert not (tmp_path / "public-release").exists()


def test_v2_probe_has_explicit_fixed_tensor_and_resource_contract():
    recipes = load_build_recipes(ROOT / 'csrc/recipes/matmul_probe.json')
    for recipe in recipes:
        assert recipe.tensor_parameter_names == ('a', 'b', 'c')
        assert tuple(p.role for p in recipe.tensor_parameters) == ('input', 'input', 'output')
        assert recipe.tiling_mode == 'host_dynamic' and recipe.tiling_schema_version == 2
        assert recipe.tiling_host_entry == 'FiMatmulProbeTiling'
        assert recipe.workspace_bytes == 0 and recipe.task_type == 'aic_only'
        assert recipe.validate_tensor_metadata(shapes={n: (16, 16) for n in ('a', 'b', 'c')},
                                              dtypes={n: 'float32' for n in ('a', 'b', 'c')}) == {}
        with pytest.raises(SchemaError, match='dimension 16'):
            recipe.validate_tensor_metadata(shapes=dict(a=(32, 16), b=(16, 16), c=(32, 16)),
                                            dtypes={n: 'float32' for n in ('a', 'b', 'c')})
        if recipe.implementation == 'native':
            resolve_build_recipe(recipe, source_root=ROOT, toolchain_fingerprint='a' * 64)


def test_public_probe_uses_common_verified_tensor_runner_and_requires_spec():
    from flashinfer.jit.tensor_adapter import TorchOpPackageRunner
    recipe = load_build_recipes(ROOT / 'csrc/recipes/matmul_probe.json')[0]
    spec = JitSpec(name='public_probe', domain='test', generator_id='ascendc', generator_version='1',
        target_soc=recipe.target_soc, environment_fingerprint='a' * 64, specialization_fingerprint='b' * 64,
        kernel_id=recipe.kernel_id, source=recipe.source, output_path='op-package.json',
        entry_points=(recipe.host_entry,), input_fingerprints=(('build_recipe', recipe.fingerprint),
                                                             ('output_adapter', OP_PACKAGE_ADAPTER_HASH)))
    module = OpPackageModule(recipe, Path('/not-loaded'), object())
    declaration = matmul_probe_op(recipe, module, spec=spec)
    assert isinstance(declaration.device_impl, TorchOpPackageRunner)
    assert declaration.mutates_args == ('out',)
    with pytest.raises(SchemaError, match='verified spec'):
        matmul_probe_op(recipe, module)
    with pytest.raises(SchemaError, match='v2 ACLNN'):
        matmul_probe_op(replace(recipe, launch_abi='aclnn.matmul_probe.v1'), module, spec=spec)


@pytest.fixture(scope='module')
def native_adapter(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    assert torch.npu.is_available() and torch.npu.device_count() == 8
    root = tmp_path_factory.mktemp('v2-probe-native-adapter')
    build_adapter(root)
    return root


@pytest.mark.real_compile
@pytest.mark.parametrize('implementation', ['native', 'catlass'])
@pytest.mark.requires_soc("Ascend950")
def test_public_v2_probe_compile_fake_namespace_and_numerical_execution(tmp_path, implementation, native_adapter):
    environment = dict(os.environ, FLASHINFER_ACLNN_ADAPTER_PATH=str(native_adapter))
    if implementation == 'catlass':
        catlass = environment['FLASHINFER_CATLASS_ROOT']
    else:
        catlass = str(tmp_path / 'absent-catlass')
    command = [sys.executable, '-B', 'tools/check_ascendc_torch_library.py', '--config',
               'examples/ascendc_' + implementation + '.json', '--cache-dir', str(tmp_path / 'cache'),
               '--catlass-root', catlass]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMPILE PASS output=op_package' in result.stdout
    assert 'NAMESPACE PASS' in result.stdout
    assert 'calls=100 before=(0, 0, 0, 0) after=(0, 0, 0, 0)' in result.stdout
    assert 'POSITIVE_CONTROL PASS' in result.stdout
    assert 'ACLNN_BRIDGE PASS native=True' in result.stdout
