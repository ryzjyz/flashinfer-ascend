"""Runner skeleton for the DS4.1 MLA prolog native contract.

The tensor order is the NativeBinding order: eleven inputs (``kv_cache`` is an
input that the kernel updates in place) followed by ``q_output``. The shared
package adapter owns device checks, layout/contiguity checks, output overlap and
the ACLNN call. This runner only preserves the cache aliasing semantics.
"""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


@dataclass(frozen=True)
class MlaPrologRunner:
    native: object

    def __call__(self, x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
                 rope_cos, rope_sin, slot_indices, kv_cache, q_output,
                 **attributes):
        if attributes:
            raise ValueError("MlaProlog has no scalar attributes")
        # kv_cache is deliberately passed as an input: the kernel scatters the
        # computed rows into the caller's storage. q_output is a fresh output.
        return self.native(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, q_output,
        )


def make_runner(module, spec, recipe):
    return MlaPrologRunner(package_tensor_runner(module, recipe, spec=spec))
