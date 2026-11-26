"""Derived capability bindings for schema-based direct operator interfaces."""

from flashinfer.runtime import Backend, KernelCapabilityBinding, SchemaError
from flashinfer.jit.core import _canonical_hash


def direct_operator_binding(operator, capability, backend, kernel_id):
    if capability.op != operator.op:
        raise SchemaError("direct capability binding requires the operator's capability")
    if operator.torch_declarations is not None:
        # Handwritten facades can mutate inputs. Bind the same functional/out
        # schemas and write aliases that will be registered with Torch.
        declarations = tuple(operator.torch_declarations())
        name = "flashinfer::" + operator.name
        names = tuple(d.name for d in declarations)
        if len(set(names)) != len(names) or name not in names or set(names) - {name, name + "_out"}:
            raise SchemaError("custom Torch declarations must identify the direct operator")
        schema = [(d.name, d.schema, d.mutates_args) for d in declarations]
    elif operator.schema and operator.public_entry is None:
        schema = operator.schema
    else:
        raise SchemaError("direct capability binding requires an exported operator schema")
    return KernelCapabilityBinding(
        domain="operator", profile_id=operator.op, rule_id=kernel_id,
        profile_fingerprint=_canonical_hash(dict(
            schema=schema, outputs=operator.output_names,
            backend=Backend(backend).value, capability=capability.to_dict(),
        )),
    )


def rich_operator_binding(operator, capability, backend, kernel_id):
    """Bind a rich public_entry operator that owns its mutation schema."""
    contract = getattr(operator, "torch_contract", ())
    if operator.schema or operator.public_entry is None or capability.op != operator.op:
        raise SchemaError("rich capability binding requires a public_entry operator")
    payload = dict(
        public_entry=operator.public_entry,
        backend=Backend(backend).value, capability=capability.to_dict(),
    )
    if contract:
        payload["torch_schema"] = contract[0]
        payload["mutates_args"] = tuple(contract[1])
    return KernelCapabilityBinding(
        domain="operator", profile_id=operator.op, rule_id=kernel_id,
        profile_fingerprint=_canonical_hash(payload),
    )


def validate_direct_operator_binding(descriptor):
    """Keep direct Tensor/ACLNN contracts distinct from AttentionPlan POD ABIs."""
    from .catalog import definition

    binding = descriptor.capability_binding
    if binding is None or binding.domain != "operator":
        raise SchemaError("direct operator requires its capability binding")
    operator = definition(descriptor.op)
    expected = (
        direct_operator_binding(operator, descriptor.capability,
                                descriptor.backend, descriptor.kernel_id)
        if operator.schema or operator.torch_declarations is not None else
        rich_operator_binding(operator, descriptor.capability,
                              descriptor.backend, descriptor.kernel_id)
    )
    if binding != expected:
        raise SchemaError("direct operator capability/schema binding mismatch")
    if descriptor.backend == Backend.ASCENDC:
        abi = descriptor.launch_abi
        if (abi is None or not abi.abi_name.startswith("aclnn.")
                or abi.argument_names != ("workspace", "workspace_bytes", "executor", "stream")):
            raise SchemaError("direct Ascend C operator requires its ACLNN launch ABI")
