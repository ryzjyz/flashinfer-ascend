"""Device accuracy for the migrated DsaSparseProlog cases."""

import pytest

import flashinfer
from flashinfer.operators.attention.dsa_sparse_prolog import cases

pytestmark = [
    pytest.mark.real_compile,
    pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
    pytest.mark.usefixtures("aot_runtime"),
]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS, ids=lambda item: item["case_name"])
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.dsa_sparse_prolog(*args, **options)
    for output, reference, (atol, rtol) in zip(actual, expected, cases.TOLERANCES):
        torch.testing.assert_close(output.cpu().double(), reference.double(), atol=atol, rtol=rtol)
    ok, rows = cases.compare(actual, configuration)
    assert ok, rows
