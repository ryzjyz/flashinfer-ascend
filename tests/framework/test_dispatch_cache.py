"""Decision cache isolation, invalidation and registry lifetime."""

from dataclasses import replace
import gc
import weakref

import pytest

from flashinfer.runtime import HardwareSignature, InputSignature
from flashinfer.runtime.dispatch import clear_dispatch_cache, dispatch
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import Backend, CapabilitySpec, KernelDescriptor


_SIGNATURE = InputSignature(op="test.dispatch_cache", dtypes=("float16",))
_HARDWARE = HardwareSignature(soc_model="Ascend910B", aicore_count=20)


def _descriptor(kernel_id, backend=Backend.CANN, priority=0):
    return KernelDescriptor(
        kernel_id=kernel_id,
        backend=backend,
        capability=CapabilitySpec(op=_SIGNATURE.op),
        priority=priority,
    )


def test_decision_cache_isolated_between_registries():
    ra = KernelRegistry([_descriptor("cann/probe/a")])
    rb = KernelRegistry([_descriptor("cann/probe/b")])
    assert ra.revision == rb.revision == 1

    # Keep both registries alive and never clear between dispatches. This also
    # runs directly, without pytest or conftest's cache-cleaning fixture.
    first_a = dispatch(_SIGNATURE, _HARDWARE, registry=ra)
    first_b = dispatch(_SIGNATURE, _HARDWARE, registry=rb)
    assert first_a.kernel_id == "cann/probe/a"
    assert first_b.kernel_id == "cann/probe/b"
    assert not first_a.cache_hit
    assert not first_b.cache_hit
    assert tuple(report.kernel_id for report in first_a.reports) == ("cann/probe/a",)
    assert tuple(report.kernel_id for report in first_b.reports) == ("cann/probe/b",)

    again_a = dispatch(_SIGNATURE, _HARDWARE, registry=ra)
    again_b = dispatch(_SIGNATURE, _HARDWARE, registry=rb)
    assert again_a == replace(first_a, cache_hit=True)
    assert again_b == replace(first_b, cache_hit=True)


@pytest.mark.parametrize("entry, backend, kernel_id", [
    ("register", Backend.CANN, "cann/probe/new"),
    ("register_external", Backend.EXTERNAL, "probe-external/probe/new"),
])
def test_registry_revision_invalidates_its_cached_decision(entry, backend, kernel_id):
    registry = KernelRegistry([_descriptor("cann/probe/base")])
    first = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    assert first.kernel_id == "cann/probe/base"
    assert not first.cache_hit
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    getattr(registry, entry)(_descriptor(kernel_id, backend, priority=10))
    assert registry.revision == 2
    updated = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    assert updated.kernel_id == kernel_id
    assert not updated.cache_hit
    assert updated.fallback_chain == (first.kernel_id,)
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry) == replace(
        updated, cache_hit=True,
    )


def test_cache_releases_registry_and_its_decisions():
    registry = KernelRegistry([_descriptor("cann/probe/transient")])
    decision = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    registry_ref = weakref.ref(registry)
    decision_ref = weakref.ref(decision)
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    del registry, decision
    gc.collect()
    assert registry_ref() is None
    # Check that dead registries do not leave cache entries accumulating either.
    assert decision_ref() is None


def test_clear_dispatch_cache_clears_all_registries():
    registries = [
        KernelRegistry([_descriptor("cann/probe/a")]),
        KernelRegistry([_descriptor("cann/probe/b")]),
    ]
    for registry in registries:
        dispatch(_SIGNATURE, _HARDWARE, registry=registry)
        assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    clear_dispatch_cache()
    for registry in registries:
        decision = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
        assert decision.kernel_id == registry.descriptors()[0].kernel_id
        assert not decision.cache_hit
        assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit


def test_operator_profiles_follow_dispatch_policy_scope_and_clear(monkeypatch):
    from flashinfer.runtime import get_global_registry, kernel_registry_scope
    from flashinfer.runtime.dispatch import (
        admit_operator_profile, operator_runner_profiles, remember_operator_runner,
    )
    from flashinfer.tuning.store import reload_tuning_records

    impl = replace(_descriptor("cann/probe/profile"), runner=lambda: None)
    registry = KernelRegistry([impl])
    registry.freeze()
    other = KernelRegistry([replace(impl, runner=lambda: 1)])
    other.freeze()
    selector = (_SIGNATURE.op, None, None)
    profile = ("float16", (16,))
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "cann")
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "0")
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", "external,cann,aot,jit,reference")

    with kernel_registry_scope(registry):
        def current_profiles():
            return operator_runner_profiles(get_global_registry(), selector, _HARDWARE)

        def install():
            decision = dispatch(_SIGNATURE, _HARDWARE, backend=None)
            entries = admit_operator_profile(registry, selector, _HARDWARE, _SIGNATURE, decision)
            assert entries is not None
            remember_operator_runner(entries, profile, impl.runner)
            assert current_profiles().get(profile) is impl.runner
            return entries

        install()
        assert operator_runner_profiles(registry, selector, HardwareSignature("Ascend950", 32)) is None
        with kernel_registry_scope(other):
            assert current_profiles() is None
        for key, value in (("FLASHINFER_NPU_BACKEND", "auto"),
                           ("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1"),
                           ("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "1"),
                           ("FLASHINFER_NPU_JIT_DISABLE", "1"),
                           ("FLASHINFER_NPU_DISPATCH_ORDER", "aot,cann,external,jit,reference"),
                           ("FLASHINFER_NPU_TUNING_DB", "/alternate/tuning.sqlite3")):
            with monkeypatch.context() as patch:
                patch.setenv(key, value)
                assert current_profiles() is None
        registry.register_external(_descriptor("peer/new", Backend.EXTERNAL))
        assert current_profiles() is None
        install()
        reload_tuning_records()
        assert current_profiles() is None
        last_entries = install()
        clear_dispatch_cache()
        # A launch that started before clear cannot republish a stale runner.
        remember_operator_runner(last_entries, profile, impl.runner)
        assert current_profiles() is None


def test_operator_profiles_are_bounded_and_released_with_the_registry(monkeypatch):
    from flashinfer.runtime.dispatch import _MAX_RUNNER_PROFILES, admit_operator_profile, remember_operator_runner

    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0")
    registry = KernelRegistry()
    # A callable can legitimately hold its own registry.
    runner = lambda registry=registry: registry
    impl = replace(_descriptor("cann/probe/bounded"), runner=runner)
    registry.register(impl)
    decision = dispatch(_SIGNATURE, _HARDWARE, backend="cann", registry=registry)
    profiles = admit_operator_profile(registry, (_SIGNATURE.op, "cann", None),
                                      _HARDWARE, _SIGNATURE, decision)
    for shape in range(_MAX_RUNNER_PROFILES + 1):
        remember_operator_runner(profiles, shape, runner)
    assert tuple(profiles) == tuple(range(1, _MAX_RUNNER_PROFILES + 1))
    registry_ref = weakref.ref(registry)
    del registry, impl, runner, profiles
    gc.collect()
    assert registry_ref() is None


def test_preparing_a_scoped_aot_runner_preserves_its_resolution_override():
    from flashinfer.jit.runtime import AotKernelRunner
    from flashinfer.runtime.runner import selected_runner

    class ScopedRunner(AotKernelRunner):
        def resolve(self):
            return lambda: "scoped"

    runner = ScopedRunner.__new__(ScopedRunner)
    impl = replace(_descriptor("ascendc/probe/scoped", Backend.ASCENDC), runner=runner)
    selected = selected_runner(impl.kernel_id, KernelRegistry([impl]), prepare=True)
    assert selected is runner
    assert selected() == "scoped"


def test_selection_options_snapshot_mutable_attributes_without_reading_opaque_values():
    from flashinfer.runtime.runner import selection_options

    class Opaque:
        def __str__(self):
            pytest.fail("forming a runner key must not inspect an opaque value")

    sizes = [16, 32]
    original = selection_options((("sizes", sizes), ("enabled", False)))
    sizes[0] = 8
    assert original == (("enabled", "False"), ("sizes", "[16, 32]"))
    assert selection_options((("sizes", sizes),)) == (("sizes", "[8, 32]"),)
    assert selection_options((("option", Opaque()),)) is None
