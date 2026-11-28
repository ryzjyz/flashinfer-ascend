"""Check the exported default route and each declared backend with independent CPU goldens."""

from contextlib import nullcontext
from importlib import import_module

import pytest

import flashinfer

from flashinfer.operators.catalog import cases_module, definition, entries
from flashinfer.operators.implementation import implementations
from flashinfer.operators.pytest_support import aot_params, implementation_param
from flashinfer.operators.testing import (
    case_output_count,
    case_parameters,
    case_tolerances,
    comparison_values,
    is_void_mutation,
    numerical_cases,
    run_void_mutation,
)
from flashinfer.runtime import Backend, CompileMode, KernelRegistry, get_global_registry, kernel_registry_scope


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
def test_signature_reads_current_storage_formats(npu_runtime, aot_runtime, monkeypatch):
    from unittest.mock import Mock
    from flashinfer.aclnn._native import load_adapter
    from flashinfer.runtime import signature

    torch, torch_npu = npu_runtime
    load_adapter()
    assert signature._batch_format_query == (torch, torch.ops.flashinfer_aclnn._storage_formats.default)
    formats = Mock(wraps=signature._batch_format_query[1])
    monkeypatch.setattr(signature, "_batch_format_query", (torch, formats))
    x = torch.empty((32, 32), dtype=torch.float16, device="npu")
    weight = torch.nn.Parameter(torch.empty_like(x), requires_grad=False)
    operands = (x, weight)
    before = signature.build_signature("probe", operands)
    formats.assert_called_once_with(operands)
    torch_npu.npu_format_cast_(x, 29)
    after = signature.build_signature("probe", operands)
    assert formats.call_count == 2
    assert before.shapes == after.shapes and before.dtypes == after.dtypes
    assert before.physical_layouts == ("ND", "ND")
    assert after.physical_layouts == ("NZ", "ND")
    assert signature.build_signature("probe", (x, None, weight)).physical_layouts == ("NZ", "", "ND")
    assert formats.call_count == 2


def _public_cases():
    for param in aot_params():
        impl, = param.values
        for index, configuration in enumerate(case_parameters(cases_module(impl.op), impl.kernel_id)):
            yield pytest.param(impl.op, configuration, marks=param.marks, id=f"{param.id}-case{index}")


@pytest.mark.parametrize("op,configuration", _public_cases())
def test_public_default_route(npu_runtime, aot_runtime, op, configuration):
    torch, torch_npu = npu_runtime
    operator = definition(op)
    if operator.public_entry:
        module, name = operator.public_entry.split(":")
        call = getattr(flashinfer, name, getattr(import_module(module), name))
    else:
        call = getattr(flashinfer, operator.name)
    cases = cases_module(op)
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    if is_void_mutation(operator, cases):
        run_void_mutation(torch, call, cases, args, options, expected)
        return
    count = case_output_count(operator, cases, configuration)
    actual = call(*args, **options)
    tolerances = case_tolerances(cases, configuration)[:count]
    if count == 1:
        actual, expected = (actual,), (expected,)
    else:
        actual, expected = tuple(actual)[:count], tuple(expected)[:count]
    assert len(actual) == len(expected) == len(tolerances)
    for index, (output, golden, (atol, rtol)) in enumerate(zip(actual, expected, tolerances)):
        if operator.schema:
            assert torch_npu.get_npu_format(output) == 2  # Typed public calls allocate ND outputs.
        if atol == rtol == 0:
            assert output.dtype == golden.dtype
        output, golden = comparison_values(cases, index, output, golden)
        torch.testing.assert_close(output, golden, atol=atol, rtol=rtol)


def _cases():
    with_samples = {entry["op"] for entry in entries() if entry["cases"]}
    return [implementation_param(impl) for impl in implementations(scope=None)
            if impl.backend != Backend.ASCENDC and impl.op in with_samples]


@pytest.mark.parametrize("impl", _cases())
def test_numerical_cases_by_kernel_id(npu_runtime, impl, monkeypatch):
    torch, torch_npu = npu_runtime
    descriptor = impl.bind()
    if descriptor.compile_mode == CompileMode.JIT:
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    scope = nullcontext()
    if impl.scope == "experimental":
        registry = KernelRegistry((*get_global_registry().descriptors(), descriptor))
        registry.freeze()
        scope = kernel_registry_scope(registry)
    with scope:
        numerical_cases(torch, torch_npu, impl.op, kernel_id=impl.kernel_id, backend=impl.backend.value)
