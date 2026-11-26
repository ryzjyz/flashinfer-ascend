"""Send CANN library operators through the same compiled call site as a custom package."""

import ctypes as C
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from csrc.cmake.operator.aclnn_adapter import fingerprint, torch_call
from flashinfer.aclnn import _native
from flashinfer.aclnn.build import build_package_call
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import TensorParameter

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.real_compile


@pytest.fixture(scope="module")
def bind(tmp_path_factory, device_runtime):
    root = tmp_path_factory.mktemp("aclnn-signatures")
    api = C.CDLL(str(Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libopapi.so"))

    def compile_call(name, inputs, attributes=(), outputs=("out",), vendor=None):
        def parameter(arg, role):
            if isinstance(arg, TensorParameter):
                return arg
            return TensorParameter(arg, role, "float32", ())
        tensors = [parameter(arg, "input") for arg in inputs] + [parameter(arg, "output") for arg in outputs]
        attrs = [AttributeParameter(*a) for a in attributes]
        raw = dict(host_entry=name, tensor_parameters=[t.to_dict() for t in tensors],
                   attributes=[a.to_dict() for a in attrs])
        source = root / (name + ".cpp")
        source.write_text(torch_call(raw))
        library = source.with_suffix(".so")
        build_package_call(source, library, ROOT)
        recipe = SimpleNamespace(fingerprint=fingerprint(raw), host_entry=name, tensor_parameters=tensors, attributes=attrs)
        module = SimpleNamespace(api=vendor or api, call_path=library)
        run = _native.prepare(module, recipe)

        def submit(tensors, *values):
            assert len(values) == len(attrs)
            return run(tensors, [a.validate(v) for a, v in zip(attrs, values)])
        return submit

    return compile_call, api


def tensor_list(name, role):
    return TensorParameter(name, role, "float32", (), tensor_list=True)


def test_tensor_lists_in_both_directions_and_integer_array(bind, device_runtime):
    torch, _ = device_runtime
    compile_call, _ = bind
    cat = compile_call("aclnnCat", [tensor_list("tensors", "input")], [("dim", "int64", 0)])
    split = compile_call("aclnnSplitWithSize", ["self"], [("splitSize", "int_array", ()), ("dim", "int64", 0)],
                         [tensor_list("out", "output")])
    for widths in ((2, 3), (1, 3, 2)):
        parts = [torch.arange(n, dtype=torch.float32) + i for i, n in enumerate(widths)]
        joined = torch.empty(sum(widths), device="npu")
        restored = [torch.empty_like(t, device="npu") for t in parts]
        cat(([v.npu() for v in parts], joined), 0)
        split((joined, restored), widths, 0)
        torch.testing.assert_close(joined.cpu(), torch.cat(parts))
        for expected, value in zip(parts, restored):
            torch.testing.assert_close(value.cpu(), expected)


def test_scalar_and_int_array(bind, device_runtime):
    torch, _ = device_runtime
    compile_call, _ = bind
    pad = compile_call("aclnnConstantPadNd", ["self"], [("pad", "int_array", ()), ("value", "scalar", 0)])
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = torch.empty((2, 6), device="npu")
    pad((x.npu(), out), (1, 2), -2.5)
    torch.testing.assert_close(out.cpu(), torch.nn.functional.pad(x, (1, 2), value=-2.5))


def test_scalar_type_selects_cast(bind, device_runtime):
    torch, _ = device_runtime
    compile_call, _ = bind
    cast = compile_call("aclnnCast", ["self"], [("dtype", "dtype", "float32")])
    x = torch.tensor([-1, 0, 1, 4], device="npu", dtype=torch.int32)
    for dtype in (torch.int8, torch.int16, torch.float16, torch.float32, torch.bfloat16, torch.bool):
        out = torch.empty_like(x, dtype=dtype)
        cast((x, out), dtype)
        torch.testing.assert_close(out.cpu(), x.cpu().to(dtype))


def test_float_array_and_string(bind, device_runtime):
    torch, torch_npu = device_runtime
    compile_call, _ = bind
    resize = compile_call("aclnnResize", ["self"], [("scales", "float_array", ()), ("mode", "string", "nearest")])
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    out = torch.empty((1, 1, 4, 6), device="npu")
    out = torch_npu.npu_format_cast(out, 2)
    resize((torch_npu.npu_format_cast(x.npu(), 2), out), (1.0, 1.0, 2.0, 2.0), "nearest")
    torch.testing.assert_close(out.cpu(), torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


def test_bool_array_and_two_outputs(bind, device_runtime):
    torch, torch_npu = device_runtime
    compile_call, _ = bind
    backward = compile_call("aclnnGridSampler2DBackward", ["gradOutput", "input", "grid"], [
        ("interpolationMode", "int64", 0), ("paddingMode", "int64", 0),
        ("alignCorners", "bool", False), ("outputMask", "bool_array", (True, True))], ["inputGrad", "gridGrad"])
    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_()
    grid = torch.tensor([[[[-0.3, 0.1], [0.25, -0.2]]]], requires_grad=True)
    result = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    grad = torch.ones_like(result)
    result.backward(grad)
    dx, dg = torch.empty_like(x, device="npu"), torch.empty_like(grid, device="npu")
    grad, xn, gn, dx, dg = (torch_npu.npu_format_cast(t, 2) for t in (grad.npu(), x.detach().npu(), grid.detach().npu(), dx, dg))
    backward((grad, xn, gn, dx, dg), 0, 0, False, (True, True))
    torch.testing.assert_close(dx.cpu(), x.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dg.cpu(), grid.grad, atol=1e-5, rtol=1e-5)


def test_host_descriptors_all_mapped_dtypes_and_optional_arguments(bind, device_runtime):
    """Inspect actual CANN descriptors at stage one, then execute a real Cast at stage two."""
    torch, _ = device_runtime
    compile_call, api = bind
    from flashinfer.jit.recipe import _TENSOR_DTYPES
    expected_types = {
        "float32": 0, "float16": 1, "int8": 2, "int32": 3, "uint8": 4, "int16": 6,
        "uint16": 7, "uint32": 8, "int64": 9, "uint64": 10, "float64": 11, "bool": 12,
        "complex64": 16, "complex128": 17, "bfloat16": 27, "int4": 29, "uint1": 30,
        "complex32": 33, "float8_e5m2": 35, "float8_e4m3fn": 36,
        "float8_e8m0fnu": 37, "float4_e2m1fn_x2": 40,
    }
    assert set(expected_types) == _TENSOR_DTYPES
    pointer = C.c_void_p
    api.aclGetDataType.argtypes = [pointer, C.POINTER(C.c_int)]
    api.aclGetRawTensorAddr.argtypes = [pointer, C.POINTER(pointer)]
    for fn in (api.aclGetTensorListSize, api.aclGetIntArraySize):
        fn.argtypes = [pointer, C.POINTER(C.c_uint64)]
    api.aclnnCastGetWorkspaceSize.argtypes = [pointer, C.c_int, pointer, pointer, pointer]
    observed = []

    def length(get, handle):
        if not handle:
            return None
        value = C.c_uint64()
        return value.value if get(handle, C.byref(value)) == 0 else -1

    @C.CFUNCTYPE(C.c_int, *([pointer] * 8))
    def workspace(x, host, tensor_list_handle, dims, scalar, out, size, executor):
        dtype, addr = C.c_int(), pointer()
        if api.aclGetDataType(host, C.byref(dtype)) or api.aclGetRawTensorAddr(host, C.byref(addr)):
            return 1
        observed.append((dtype.value, addr.value, length(api.aclGetTensorListSize, tensor_list_handle),
                         length(api.aclGetIntArraySize, dims), bool(scalar)))
        return api.aclnnCastGetWorkspaceSize(x, 0, out, size, executor)

    vendor = SimpleNamespace(aclnnInspectHostGetWorkspaceSize=workspace, aclnnInspectHost=api.aclnnCast)
    inputs = ["x", TensorParameter("host", "input", "float32", (8,), device="cpu"),
              TensorParameter("extras", "input", "float32", (4,), optional=True, tensor_list=True)]
    run = compile_call("aclnnInspectHost", inputs, [("dims", "int_array", None, None, None, True),
                       ("value", "scalar", None, None, None, True)], vendor=vendor)
    x = torch.arange(4, dtype=torch.float16, device="npu")
    for i, (name, expected) in enumerate(expected_types.items()):
        host = torch.empty(8, dtype=getattr(torch, name))
        out = torch.empty_like(x, dtype=torch.float32)
        extras = (None, [], [x, x])[i % 3]
        dims = (None, (), (0, 1))[i % 3]
        scalar = (None, -2.5, 1j)[i % 3]
        run((x, host, extras, out), dims, scalar)
        assert observed[-1] == (expected, host.data_ptr(), None if extras is None else len(extras),
                                None if dims is None else len(dims), scalar is not None)
        torch.testing.assert_close(out.cpu(), x.cpu().float())
