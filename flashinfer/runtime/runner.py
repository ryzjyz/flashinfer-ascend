"""Select and reuse eager operator runners without retaining live call arguments."""

from functools import partial
import sys

from .. import config
from . import signature as signatures
from .dispatch import admit_operator_profile, dispatch, operator_runner_profiles, remember_operator_runner
from .hardware import current_hardware
from .registry import get_global_registry
from .schema import DispatchError


def selected_runner(kernel_id: str, registry=None, *, prepare=False):
    """Get the callable; a prepared AOT call can reuse its already bound package."""
    descriptor = (registry if registry is not None else get_global_registry()).get(kernel_id)
    runner = descriptor.runner
    if runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入" % kernel_id
        )
    jit = sys.modules.get("flashinfer.jit.runtime")
    if prepare and jit is not None:
        AotKernelRunner = jit.AotKernelRunner
        if isinstance(runner, AotKernelRunner) and type(runner).__call__ is AotKernelRunner.__call__:
            resolve = runner.resolve
            if getattr(resolve, "__func__", None) is AotKernelRunner.resolve:
                return resolve()
    return runner


def run_selected(kernel_id: str, *args, **kwargs):
    """Legacy or custom frontends may already have selected a concrete kernel."""
    registry = get_global_registry()
    profile = None if config.dispatch_cache_disabled() else _argument_profile(args, kwargs)
    if profile is None:
        return selected_runner(kernel_id, registry)(*args, **kwargs)
    selector = (None, "auto", kernel_id)
    hardware = current_hardware()
    profiles = operator_runner_profiles(registry, selector, hardware)
    if profiles is not None:
        hit = profiles.get(profile)
        if hit is not None:
            return hit[1](*args, **kwargs)
    profiles = admit_operator_profile(registry, selector, hardware, None, None)
    return _first_runner(kernel_id, registry, profiles, profile)(*args, **kwargs)


def _first_runner(kid, registry, profiles, profile):
    def first_launch(*operands, **attrs):
        # Keep package loading after all per-invocation validation.
        runner = selected_runner(kid, registry, prepare=profiles is not None)
        result = runner(*operands, **attrs)
        if profiles is not None:
            remember_operator_runner(profiles, profile, (kid, runner))
        return result
    return first_launch


class _Uncacheable(Exception):
    """A call that must follow normal dispatch."""


def _plain_option(value):
    return (value is None or type(value) in (str, bool, int, float)
            or type(value) in (tuple, list) and all(map(_plain_option, value)))


def selection_options(items):
    """Snapshot simple specializations exactly as build_signature does, without tensor reads."""
    result = []
    try:
        for name, value in items:
            if type(name) is not str or not _plain_option(value):
                return None
            result.append((name, str(value)))
    except RecursionError:
        return None
    return tuple(sorted(result))


def _argument_key(value, torch, plain_types, tensors):
    kind = type(value)
    if kind in plain_types:
        tensors.append(value)
        return kind
    if value is None or kind in (bool, int, str, torch.dtype, torch.device):
        return (kind, value)
    if kind is float:
        return (kind, value.hex())
    if kind in (tuple, list):
        return (kind, tuple(_argument_key(v, torch, plain_types, tensors) for v in value))
    if kind is dict and all(type(k) is str for k in value):
        return (kind, tuple((k, _argument_key(value[k], torch, plain_types, tensors))
                            for k in sorted(value)))
    raise _Uncacheable


def _argument_profile(args, attributes):
    """Snapshot NPU metadata, optional/container structure and typed scalar values."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    plain_types = (torch.Tensor, torch.nn.Parameter)
    try:
        if not attributes and args and all(type(v) in plain_types for v in args):
            tensors = args
            structure = ()
        else:
            tensors = []
            operands = tuple(_argument_key(v, torch, plain_types, tensors) for v in args)
            attrs = _argument_key(attributes, torch, plain_types, tensors) if attributes else ()
            structure = (operands, attrs)
            if not tensors:
                return None
        device = tensors[0].device
        if (device.type not in ("npu", "privateuseone") or torch.npu.current_device() != device.index
                or torch.overrides.has_torch_function(tensors)
                or torch.utils._python_dispatch.is_in_torch_dispatch_mode()
                or torch.npu.is_current_stream_capturing()
                or any(t.device != device or t.layout != torch.strided for t in tensors)):
            return None
        metadata = tuple((type(t), t.dtype, t.shape, t.stride()) for t in tensors)
        batch = signatures._batch_format_query
        if batch is not None:
            formats = tuple(batch[1](tensors))
        else:
            npu = sys.modules.get("torch_npu")
            if npu is None:
                return None
            formats = tuple(int(npu.get_npu_format(t)) for t in tensors)
    except (_Uncacheable, RuntimeError, RecursionError):
        return None
    return device, structure, metadata, formats


def resolve_operator_runner(op, args, attributes, request, *, backend, kernel_id, hardware,
                            facts=(), select=None, fallback=run_selected):
    """Return (kernel ID, callable). A callable request builds the full signature on a miss."""
    registry = get_global_registry()
    selector = (op, backend, kernel_id)
    valid_selector = ((backend is None or isinstance(backend, str))
                      and (kernel_id is None or isinstance(kernel_id, str)))
    profile = (_argument_profile(args, attributes)
               if facts is not None and valid_selector and not config.dispatch_cache_disabled() else None)
    if profile is not None:
        profile = (profile, facts, None if callable(request) else request)
        profiles = operator_runner_profiles(registry, selector, hardware)
        if profiles is not None:
            hit = profiles.get(profile)
            if hit is not None:
                return hit
    signature = request() if callable(request) else request
    select = dispatch if select is None else select
    decision = select(signature, hardware, backend=backend, kernel_id=kernel_id, registry=registry)
    kid = decision.kernel_id
    profiles = (admit_operator_profile(registry, selector, hardware, signature, decision)
                if profile is not None else None)
    if profiles is None:
        if fallback is run_selected:
            return kid, _first_runner(kid, registry, None, None)
        return kid, partial(fallback, kid)

    return kid, _first_runner(kid, registry, profiles, profile)


def run_operator(op, tensors, args=None, *, backend=None, kernel_id=None, attributes=None,
                 specializations=None, layouts=None, inference=None, hardware=None):
    """Common handoff for domain frontends using the standard signature builder."""
    tensors = tuple(tensors)
    args = tensors if args is None else tuple(args)
    profile_args = args
    routing_positions = ()
    if args is not tensors:
        # Legacy ABIs can select on a subset (or a different order) of launch operands.
        # Positions are sufficient when their metadata is already in the launch profile.
        by_id = {id(arg): index for index, arg in enumerate(args)}
        routing_positions = tuple(by_id.get(id(t), -1) for t in tensors)
        if -1 in routing_positions:
            profile_args = (args, tensors)
    layouts = None if layouts is None else tuple(layouts)

    def request():
        return signatures.build_signature(op, tensors, layouts=layouts,
                                          specializations=specializations, inference=inference)

    hardware = current_hardware() if hardware is None else hardware
    options = selection_options(specializations.items()) if specializations else ()
    facts = (routing_positions, layouts, options, inference) if options is not None else None
    _, runner = resolve_operator_runner(op, profile_args, attributes, request, hardware=hardware,
                                        backend=backend, kernel_id=kernel_id, facts=facts)
    return runner(*args, **(attributes or {}))
