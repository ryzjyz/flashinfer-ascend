"""Independent CPU scoring and deterministic IRS acceptance inputs."""

from itertools import accumulate


def golden(torch, query, weights, key_summary, *, actual_kv_lengths=None,
           block_table=None, cu_seq_lens=None, mask_mode=0, head_scale=1.0,
           cache_mode=0, max_region_len=-1, region_size=8, top_k=512,
           sort_mode=0, return_score=True):
    if sort_mode != 0:
        raise ValueError("sort_mode must be 0 (ascending region ID)")
    tokens, groups, heads, dim = query.shape
    regions = key_summary.shape[0] if max_region_len == -1 else max_region_len
    scores = torch.full((groups, tokens, regions), -float("inf"), dtype=torch.float32)
    selected = torch.full((groups, tokens, top_k), -1, dtype=torch.int32)
    boundaries = [0, tokens] if cu_seq_lens is None else cu_seq_lens.tolist()
    for request, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        keys = key_summary[:, 0].double()
        if cache_mode == 1:
            page_size = key_summary.shape[1]
            keys = torch.zeros((regions, dim), dtype=torch.float64)
            for region in range(regions):
                page = int(block_table[request, region // page_size])
                if 0 <= page < key_summary.shape[0]:
                    keys[region] = key_summary[page, region % page_size, 0].double()
        else:
            keys = keys[:regions]
        for token in range(begin, end):
            visible = regions
            if mask_mode == 3:
                position = int(actual_kv_lengths[request]) - (end - begin) + token - begin
                visible = min(regions, max(0, position // region_size))
            for group in range(groups):
                dots = query[token, group].double() @ keys[:visible].T
                # Weight scaling precedes the head sum in the native formula.
                values = (dots.clamp_min(0) * (weights[token, group].double() * head_scale)[:, None]).sum(0)
                scores[group, token, :visible] = values.float()
                valid = [index for index, value in enumerate(values.tolist()) if value > -1.0e29]
                ranked = sorted(valid, key=lambda index: (-float(values[index]), index))[:top_k]
                ranked.sort()
                selected[group, token, :len(ranked)] = torch.tensor(ranked, dtype=torch.int32)
    return (scores if return_score else torch.empty(0, dtype=torch.float32)), selected


def sample(torch, torch_npu, *, dtype="bfloat16", weight_dtype="float32", paged=False,
           causal=True, return_score=True, dim=256, heads=4, groups=1,
           request_lengths=(3, 2), regions=33, page_size=16, top_k=8,
           pitched=False, max_region_len=None):
    tokens = sum(request_lengths)
    rng = torch.Generator().manual_seed(20261124 + dim + regions + top_k)
    q = (torch.rand((tokens, groups, heads, dim), generator=rng) * 0.5 + 0.25).to(getattr(torch, dtype))
    weights = (torch.rand((tokens, groups, heads), generator=rng) + 0.5).to(getattr(torch, weight_dtype))
    # Separated positive key magnitudes make exact ID expectations stable across
    # FP8/BF16 multiplication rounding while still testing the full dot product.
    key = torch.arange(1, regions + 1, dtype=torch.float32)[:, None, None].expand(regions, 1, dim).clone()
    key = (key / 16).to(getattr(torch, dtype))
    cu = torch.tensor([0] + list(accumulate(request_lengths)), dtype=torch.int32)
    lengths = torch.tensor([regions * 8 - i * 16 + count for i, count in enumerate(request_lengths)], dtype=torch.int32)
    table = None
    if paged:
        pages_per_request = (regions + page_size - 1) // page_size
        table = torch.arange(len(request_lengths) * pages_per_request - 1, -1, -1,
                             dtype=torch.int32).reshape(len(request_lengths), pages_per_request)
        packed = torch.zeros((table.numel(), page_size, 1, dim), dtype=key.dtype)
        for request in range(len(request_lengths)):
            for region in range(regions):
                packed[table[request, region // page_size], region % page_size] = key[region]
        key = packed
    if not causal and not paged:
        cu = None
    effective_regions = regions if max_region_len is None else max_region_len
    options = dict(mask_mode=3 if causal else 0, head_scale=0.5, cache_mode=int(paged),
                   max_region_len=effective_regions, region_size=8, top_k=top_k,
                   sort_mode=0, return_score=return_score)
    cpu_options = dict(options, actual_kv_lengths=lengths if causal else None,
                       block_table=table, cu_seq_lens=cu)
    expected = golden(torch, q, weights, key, **cpu_options)
    def device(tensor):
        return None if tensor is None else torch_npu.npu_format_cast(tensor.npu(), 2)
    device_key = device(key)
    if pitched:
        backing = device(torch.zeros((key.shape[0], page_size + 16, 1, dim), dtype=key.dtype))
        device_key = backing[:, :page_size]
        device_key.copy_(device(key))
    options.update(actual_kv_lengths=device(lengths) if causal else None,
                   block_table=device(table), cu_seq_lens=device(cu))
    return (device(q), device(weights), device_key), options, expected


CASE_PARAMETERS = (
    dict(),
    dict(return_score=False),
    dict(paged=True, pitched=True, dim=128, dtype="float16"),
    dict(paged=True, pitched=True, return_score=False, dim=128, top_k=64),
    dict(paged=True, causal=False, dtype="float8_e4m3fn", regions=48, page_size=16, weight_dtype="bfloat16"),
    dict(causal=False, groups=4, heads=8, weight_dtype="float16"),
    dict(paged=True, max_region_len=0, top_k=512),
)
TOLERANCES = ((0.02, 0.003), (0, 0))


def comparison_mask(output_index, reference):
    """The CPU sample's -inf marks Score positions outside the visible range."""
    return ~reference.isneginf() if output_index == 0 else None
