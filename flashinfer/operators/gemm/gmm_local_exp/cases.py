"""Independent CPU expert-wise GEMM oracle and complete native tiling-key matrix."""


def golden(torch, x, weight, counts, start, end, *, trans_b=False, type_promotion=False):
    n = weight.shape[1 if trans_b else 2]
    output = torch.zeros(x.shape[0], n, dtype=torch.float32 if type_promotion else x.dtype)
    offset = 0
    for expert, count in enumerate(counts):
        if start <= expert < end and count:
            w = weight[expert - start]
            output[offset:offset + count] = (x[offset:offset + count].double() @
                                             (w.T if trans_b else w).double()).to(output.dtype)
        offset += count
    return output


def sample(torch, torch_npu, *, dtype="float16", trans_b=False, is_b_nz=False,
           type_promotion=False, counts=(17, 0, 9, 7), start=0, end=4, k=64, n=96):
    rng = torch.Generator().manual_seed(12993)
    x = torch.randn(sum(counts), k, generator=rng).to(getattr(torch, dtype))
    weight = torch.randn(end - start, n if trans_b else k, k if trans_b else n,
                         generator=rng).to(x.dtype)
    expected = golden(torch, x, weight, counts, start, end, trans_b=trans_b, type_promotion=type_promotion)
    if is_b_nz:
        # Preserve the source API's packed-NZ-in-ND buffer convention.
        weight = weight.reshape(end - start, n, k // 16, 16).permute(0, 2, 1, 3).contiguous().reshape_as(weight)
    args = (x.npu(), weight.npu(), torch.tensor(counts, dtype=torch.int32).npu(), start, end)
    return args, dict(trans_b=trans_b, is_b_nz=is_b_nz, type_promotion=type_promotion), expected


CASE_PARAMETERS = tuple(dict(dtype=d, trans_b=t, is_b_nz=z, type_promotion=p)
                        for d in ("float16", "bfloat16") for t, z in ((False, False), (True, False), (True, True))
                        for p in (False, True)) + (
    dict(counts=(1, 0, 513, 1), k=256, n=512),
)
TOLERANCES = ((0.125, 0.015),)


def tolerances(configuration):
    if configuration.get("type_promotion", False):
        return ((0.0002, 0.0001),)
    if configuration.get("dtype", "float16") == "float16":
        return ((0.015625, 0.002),)
    return TOLERANCES
