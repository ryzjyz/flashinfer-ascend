"""Recipe-driven eager tensor adapter for the two-stage ACLNN package ABI.

This module owns tensor metadata/lifetimes, not package discovery or loading.
Tiling is produced by the package's host callback and retained by the ACLNN
executor; its device buffer is an implicit final kernel argument.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import repeat
from threading import RLock

from flashinfer.runtime import SchemaError


def _storage_span(parameter, tensor, layout):
    if tensor is None or getattr(parameter, "device", None) == "cpu" or tensor.numel() == 0:
        return 0, 0
    itemsize = tensor.element_size()
    if layout == "NZ":
        storage = tensor.untyped_storage()
        begin = storage.data_ptr()
        return begin, begin + storage.nbytes()
    begin = tensor.data_ptr()
    if not hasattr(tensor, "stride") or tensor.is_contiguous():
        return begin, begin + tensor.numel() * itemsize
    strides = tensor.stride()
    axes = [(stride, size - 1) for size, stride in zip(tensor.shape, strides) if size > 1]
    if any(stride < 0 for stride in strides) or any(stride == 0 for stride, _ in axes):
        raise ValueError("ACLNN strided Tensor requires nonnegative strides and positive strides on non-singleton axes")
    elements = 1 + sum(stride * steps for stride, steps in axes)
    offset = tensor.storage_offset()
    if offset < 0 or offset + elements > tensor.untyped_storage().nbytes() // itemsize:
        raise ValueError("ACLNN strided Tensor exceeds its storage span")
    if parameter.role == "output":
        occupied = 1
        for stride, steps in sorted(axes):
            if stride < occupied:
                raise ValueError("ACLNN strided output has internally overlapping elements")
            occupied += stride * steps
    return begin, begin + elements * itemsize


def _reject_output_overlap(parameters, tensors, *, layouts=None):
    spans = tuple(_storage_span(p, t, layout) for p, t, layout in
                  zip(parameters, tensors, repeat(None) if layouts is None else layouts))
    aliases = {}
    if any(getattr(p, "alias_of", None) is not None for p in parameters):
        from torch._C import _is_alias_of

        indices = {p.name: i for i, p in enumerate(parameters)}
        for index, p in enumerate(parameters):
            if p.alias_of is None:
                continue
            input_index = indices[p.alias_of]
            source, target = tensors[input_index], tensors[index]
            if (source.device != target.device or not _is_alias_of(source, target)
                    or source.dtype != target.dtype or source.shape != target.shape
                    or source.stride() != target.stride() or source.storage_offset() != target.storage_offset()):
                raise ValueError("ACLNN alias pair must be the exact same storage view")
            aliases[index] = input_index
    for output_index, (parameter, output) in enumerate(zip(parameters, tensors)):
        if parameter.role != "output":
            continue
        out_lo, out_hi = spans[output_index]
        for tensor_index, tensor in enumerate(tensors):
            if (tensor_index == output_index or tensor_index == aliases.get(output_index)
                    or tensor is None or tensor.device != output.device):
                continue
            lo, hi = spans[tensor_index]
            if max(lo, out_lo) < min(hi, out_hi):
                raise ValueError("output tensors must not overlap other arguments")


def package_tensor_runner(module, recipe, *, spec):
    """Validate a previously loaded package and return its generic adapter."""
    from .loading import JitLoadedModule
    from .op_package import OpPackageModule

    if (spec.kernel_id != recipe.kernel_id or spec.source != recipe.source
            or spec.target_soc != recipe.target_soc
            or spec.entry_points != (recipe.host_entry,)
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint):
        raise SchemaError("tensor adapter spec does not bind the recipe/entry")
    if isinstance(module, JitLoadedModule):
        if (module.receipt.spec_fingerprint != spec.fingerprint
                or tuple(symbol.name for symbol in module.receipt.symbols)
                != (recipe.host_entry,)):
            raise SchemaError("tensor adapter loaded receipt does not bind its spec")
        module = module.opaque_module
    if not isinstance(module, OpPackageModule) or module.recipe != recipe:
        raise SchemaError("tensor adapter requires the verified recipe's OpPackageModule")
    return TorchOpPackageRunner(module, recipe)


@dataclass(frozen=True)
class TorchOpPackageRunner:
    module: object
    recipe: object
    _native_runner: object = field(default=None, init=False, repr=False, compare=False)
    _native_lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _validate_metadata: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        recipe = self.recipe

        @lru_cache(maxsize=128)
        def validate(metadata):
            shapes, dtypes = {}, {}
            for p, item in zip(recipe.tensor_parameters, metadata):
                if item is None:
                    shapes[p.name] = dtypes[p.name] = None
                elif p.tensor_list:
                    shapes[p.name] = tuple(t[0] for t in item)
                    dtypes[p.name] = tuple(str(t[1]).removeprefix("torch.") for t in item)
                else:
                    shapes[p.name], dtype = item
                    dtypes[p.name] = str(dtype).removeprefix("torch.")
            recipe.validate_tensor_metadata(shapes=shapes, dtypes=dtypes)

        object.__setattr__(self, "_validate_metadata", validate)

    def _validate_storage(self, tensors):
        import torch
        import torch_npu

        arguments = [(p,t) for p, value in zip(self.recipe.tensor_parameters, tensors)
                     if value is not None for t in (value if p.tensor_list else (value,))]
        npu = torch._C._get_privateuse1_backend_name()
        device = next((t.device for p,t in arguments if p.device == "npu"), None)
        if device is None or device.type != npu:
            raise ValueError("Ascend C tensor adapter requires NPU tensors")
        layouts = []
        for p, t in arguments:
            if p.device == "cpu":
                if t.device.type != "cpu":
                    raise ValueError("expected a CPU input for " + p.name)
                layouts.append(None)
                continue
            if t.device != device:
                raise ValueError("Ascend C tensors must share one device")
            layout = {2: "ND", 29: "NZ"}.get(torch_npu.get_npu_format(t))
            if layout not in p.physical_layouts or (
                    not t.is_contiguous() and not (p.allow_strided and layout == "ND")):
                raise ValueError("Ascend C tensor adapter requires contiguous %s tensors for %s"
                                 % ("/".join(p.physical_layouts), p.name))
            layouts.append(layout)
        if arguments:
            params, flat = zip(*arguments)
            _reject_output_overlap(params, flat, layouts=layouts)

    def __call__(self, *tensors, **attributes):
        names = self.recipe.tensor_parameter_names
        if len(tensors) != len(names):
            raise SchemaError("tensor adapter expected parameters %s" % (names,))
        metadata = []
        for p, value in zip(self.recipe.tensor_parameters, tensors):
            if value is None:
                metadata.append(None)
            elif p.tensor_list:
                if not isinstance(value, (tuple, list)):
                    raise SchemaError("expected TensorList for " + p.name)
                metadata.append(tuple((tuple(t.shape), t.dtype) for t in value))
            else:
                metadata.append((tuple(value.shape), value.dtype))
        self._validate_metadata(tuple(metadata))
        scalar_values = self.recipe.validate_attributes(attributes)

        if self._native_runner is None:
            self._validate_storage(tensors)
            from flashinfer.aclnn import _native

            with self._native_lock:
                if self._native_runner is None:
                    object.__setattr__(self, "_native_runner", _native.prepare(self.module, self.recipe))
        # The typed call checks storage on every invocation, including warm hits.
        try:
            return self._native_runner(tensors, scalar_values) if scalar_values else self._native_runner(tensors)
        except RuntimeError:
            # Preserve the eager adapter's diagnostic for invalid storage.
            self._validate_storage(tensors)
            raise


__all__ = ["TorchOpPackageRunner", "package_tensor_runner"]
