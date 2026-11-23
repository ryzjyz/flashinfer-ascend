"""Targeted WAR and queue-wrap regressions for dense IndexerPrologueQW."""

import pytest

import flashinfer

pytestmark = [pytest.mark.real_compile,
              pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("tokens", [65, 128])
def test_repeated_mixed_amax_and_queue_wrap(npu_runtime, tokens):
    """Repeated AIV rows with different amax must survive QW scale/q reuse.

    Each row is compared with an isolated T=1 launch of the same row; this
    avoids DynamicQuant half-way ties in a CPU oracle while still detecting a
    WAR overwrite of the scale/q UB slots.
    """
    import contextlib
    import math

    torch, torch_npu = npu_runtime
    generator = torch.Generator(device="cpu").manual_seed(20261120 + tokens)
    row_scale = (2.0 ** torch.linspace(-4.0, 4.0, tokens, dtype=torch.float32))
    weight_scale = (2.0 ** torch.linspace(-3.0, 3.0, 4096, dtype=torch.float32))
    x = torch.randn(tokens, 5120, generator=generator).to(torch.bfloat16)
    qr = (torch.randn(tokens, 1280, generator=generator) * row_scale[:, None]).to(torch.bfloat16)
    wqb = (torch.randn(4096, 1280, generator=generator) * weight_scale[:, None]
           / math.sqrt(1280)).to(torch.bfloat16)
    ww = (torch.randn(32, 5120, generator=generator) / math.sqrt(5120)).to(torch.bfloat16)
    sin, cos = torch.randn(tokens, 64, generator=generator), torch.randn(tokens, 64, generator=generator)
    args = (x.npu(), qr.npu(), wqb.npu(), ww.npu(), sin.npu(), cos.npu())
    reference_rows = (0, 31, tokens - 1)
    references = {}
    for row in reference_rows:
        with torch.no_grad():
            references[row] = tuple(
                tensor.cpu().clone() for tensor in flashinfer.indexer_prologue_qw(
                    x[row:row + 1].npu(), qr[row:row + 1].npu(), wqb.npu(), ww.npu(),
                    sin[row:row + 1].npu(), cos[row:row + 1].npu()))
    first = None
    for repeat in range(4):
        stream = torch_npu.npu.Stream() if repeat == 2 else None
        context = torch_npu.npu.stream(stream) if stream is not None else contextlib.nullcontext()
        with context:
            actual = flashinfer.indexer_prologue_qw(*args)
        if stream is not None:
            stream.synchronize()
        for row in reference_rows:
            for output, reference in zip(actual, references[row]):
                torch.testing.assert_close(output.cpu()[row].double(), reference[0].double(),
                                           atol=0.0, rtol=0.0)
        if first is None:
            first = tuple(value.cpu().clone() for value in actual)
        else:
            assert all(torch.equal(value.cpu(), prior) for value, prior in zip(actual, first))
