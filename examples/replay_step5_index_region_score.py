"""Replay the Step5 IRS shapes captured with 32 AIC / 64 AIV.

Synthetic replay uses deterministic inputs and a synthetic page table. Decode
defaults to the historical replay assumption of 72,000 KV tokens per request.
Exact replay requires a torch.save dictionary containing the six INPUT_NAMES
below and an integer max_region_len from the same model invocation.
NPU event intervals can include Python enqueue gaps. Use a task-level profile
when comparing kernel time with the model trace.

    python examples/replay_step5_index_region_score.py --case all --device 0
    python examples/replay_step5_index_region_score.py --case decode --dump step.pt
"""

import argparse
import ctypes
import json
from pathlib import Path
import statistics


INPUT_NAMES = (
    "query", "weights", "key_summary", "actual_kv_lengths", "block_table", "cu_seq_lens",
)


def synthetic_inputs(case, kv_length, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    tokens, requests, pages = (16384, 1, 21392) if case == "prefill" else (80, 20, 26378)
    regions = (kv_length - 1) // 8 if case == "prefill" else 131072
    query = torch.randn((tokens, 1, 4, 256), generator=generator).to(torch.float8_e4m3fn)
    weights = torch.randn((tokens, 1, 4), generator=generator)
    key = torch.randn((pages, 16, 1, 256), generator=generator).to(torch.float8_e4m3fn)
    # Each request has a valid contiguous logical page sequence. Wraparound
    # shares physical pages between requests within the captured cache size.
    table = (torch.arange(requests * 8192).reshape(requests, 8192) % pages).int()
    return dict(
        query=query, weights=weights, key_summary=key,
        actual_kv_lengths=torch.full((requests,), kv_length, dtype=torch.int32),
        block_table=table,
        cu_seq_lens=torch.arange(0, tokens + 1, tokens // requests, dtype=torch.int32),
    ), regions


def validate_inputs(inputs, regions):
    import torch

    if type(regions) is not int or not 0 <= regions <= 131072:
        raise ValueError("max_region_len must be an integer in [0,131072]")
    for name in INPUT_NAMES:
        if not isinstance(inputs[name], torch.Tensor) or inputs[name].device.type != "cpu":
            raise ValueError(f"{name} must be a CPU Tensor")
    q, weights, key, lengths, table, cu = (inputs[name] for name in INPUT_NAMES)
    if q.ndim != 4 or tuple(q.shape[1:]) != (1, 4, 256) or q.dtype != torch.float8_e4m3fn:
        raise ValueError("query must be FP8 E4M3FN [T,1,4,256]")
    if weights.shape != q.shape[:-1] or weights.dtype != torch.float32:
        raise ValueError("weights must be FP32 [T,1,4]")
    if key.ndim != 4 or tuple(key.shape[1:]) != (16, 1, 256) or key.dtype != q.dtype:
        raise ValueError("key_summary must be FP8 E4M3FN [P,16,1,256]")
    if table.ndim != 2 or table.shape[1] != 8192 or table.dtype != torch.int32:
        raise ValueError("block_table must be INT32 [N,8192]")
    requests = table.shape[0]
    if lengths.shape != (requests,) or lengths.dtype != torch.int32:
        raise ValueError("actual_kv_lengths must be INT32 [N]")
    if cu.shape != (requests + 1,) or cu.dtype != torch.int32:
        raise ValueError("cu_seq_lens must be INT32 [N+1]")
    query_lengths = cu[1:].long() - cu[:-1].long()
    if cu[0] != 0 or cu[-1] != q.shape[0] or (query_lengths <= 0).any():
        raise ValueError("cu_seq_lens must cover all query rows with positive request lengths")
    if (lengths.long() < query_lengths).any() or (lengths > 1048576).any():
        raise ValueError("KV lengths must cover each request's query and fit the page-table capacity")
    for request in range(requests):
        visible = min(regions, (int(lengths[request]) - 1) // 8)
        active = table[request, :(visible + 15) // 16]
        if ((active < 0) | (active >= key.shape[0])).any():
            raise ValueError(f"block_table request {request} has an invalid active physical page")


def validate_ids(ids, inputs, regions):
    """Check every row's count, padding, ascending IDs and causal region bound."""
    import torch

    ids = ids.cpu()
    tokens = inputs["query"].shape[0]
    if ids.shape != (1, tokens, 512) or ids.dtype != torch.int32:
        raise AssertionError(f"unexpected selected_region_ids {ids.shape} {ids.dtype}")
    cu = inputs["cu_seq_lens"].long()
    query_lengths = cu[1:] - cu[:-1]
    starts = inputs["actual_kv_lengths"].long() - query_lengths
    positions = torch.arange(tokens) + torch.repeat_interleave(starts - cu[:-1], query_lengths)
    visible = (positions // 8).clamp(min=0, max=regions)
    count = visible.clamp(max=512)
    expected_valid = torch.arange(512).unsqueeze(0) < count.unsqueeze(1)
    rows = ids[0]
    if not torch.equal(rows >= 0, expected_valid):
        raise AssertionError("IDs have an incorrect candidate count or padding position")
    if (rows[~expected_valid] != -1).any():
        raise AssertionError("IDs padding must be -1")
    if ((rows >= visible.unsqueeze(1)) & expected_valid).any():
        raise AssertionError("an ID exceeds its row's causal visible region bound")
    if ((rows[:, 1:] <= rows[:, :-1]) & expected_valid[:, 1:]).any():
        raise AssertionError("valid IDs must be strictly ascending")


def run_case(case, kv_length, args):
    import torch
    import flashinfer

    if args.dump:
        saved = torch.load(args.dump, map_location="cpu", weights_only=True)
        inputs = {name: saved[name] for name in INPUT_NAMES}
        regions = saved["max_region_len"]
    else:
        inputs, regions = synthetic_inputs(case, kv_length, args.seed)
    validate_inputs(inputs, regions)
    device = torch.device(f"npu:{args.device}")
    # Preserve pitched page layouts from a dump as well as dense model tensors.
    device_inputs = {}
    for name, tensor in inputs.items():
        copied = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
        copied.copy_(tensor)
        device_inputs[name] = copied
    query, weights, key = (device_inputs[name] for name in INPUT_NAMES[:3])
    kwargs = {name: device_inputs[name] for name in INPUT_NAMES[3:]}
    kwargs.update(mask_mode=3, head_scale=args.head_scale, cache_mode=1,
                  max_region_len=regions, region_size=8, top_k=512,
                  sort_mode=0, return_score=not args.ids_only, backend="ascendc")

    for _ in range(args.warmup):
        score, ids = flashinfer.index_region_score(query, weights, key, **kwargs)
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        score, ids = flashinfer.index_region_score(query, weights, key, **kwargs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    validate_ids(ids, inputs, regions)
    expected_score = (0,) if args.ids_only else (1, query.shape[0], regions)
    if tuple(score.shape) != expected_score or score.dtype != torch.float32:
        raise AssertionError(f"unexpected score output {score.shape} {score.dtype}")
    print(json.dumps(dict(
        case=case, source=str(args.dump) if args.dump else "synthetic",
        kv_lengths=inputs["actual_kv_lengths"].tolist(),
        shapes={name: list(tensor.shape) for name, tensor in inputs.items()},
        score_shape=list(score.shape), ids_shape=list(ids.shape),
        ids_contract="passed", return_score=not args.ids_only,
        event_us_p50=round(statistics.median(samples), 3), event_us_min=round(min(samples), 3),
        samples=args.repeats,
    )), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--device", type=int, default=0, help="visible NPU index")
    parser.add_argument("--allow-non-32-cores", action="store_true",
                        help="permit a shape replay on a device with a different core count")
    parser.add_argument("--prefill-kv", type=int, choices=range(16384, 131073, 16384),
                        help="select one of the eight captured prefill widths")
    parser.add_argument("--decode-kv", type=int, default=72000,
                        help="synthetic replay assumption, in tokens per request")
    parser.add_argument("--dump", type=Path, help="torch.save CPU input dictionary from one IRS call")
    parser.add_argument("--ids-only", action="store_true")
    parser.add_argument("--head-scale", type=float, default=1.0,
                        help="model weights already contain H**-0.5 scaling")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.dump and args.case == "all":
        parser.error("--dump requires --case prefill or --case decode")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup must be nonnegative")
    if not 4 <= args.decode_kv <= 1048576:
        parser.error("--decode-kv must be in [4,1048576]")

    import torch
    import torch_npu  # noqa: F401
    import flashinfer

    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable")
    torch.npu.set_device(args.device)
    torch.empty((1,), device=f"npu:{args.device}")
    runtime = ctypes.CDLL("libascendcl.so")
    get_info = runtime.aclrtGetDeviceInfo
    get_info.argtypes = (ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64))
    get_info.restype = ctypes.c_int32
    core_counts = []
    # acl_rt.h defines ACL_DEV_ATTR_CUBE_CORE_NUM=102 and VECTOR_CORE_NUM=201.
    for attribute in (102, 201):
        value = ctypes.c_int64()
        status = get_info(args.device, attribute, ctypes.byref(value))
        if status:
            raise RuntimeError(f"aclrtGetDeviceInfo({attribute}) failed with {status}")
        core_counts.append(value.value)
    if core_counts != [32, 64] and not args.allow_non_32_cores:
        raise RuntimeError(f"32 AIC / 64 AIV required, found {core_counts}. "
                           "Use --allow-non-32-cores for a shape replay on this device.")
    print(json.dumps(dict(flashinfer=flashinfer.__file__, device=args.device,
                          device_name=torch.npu.get_device_name(args.device),
                          aic_cores=core_counts[0], aiv_cores=core_counts[1])), flush=True)
    if args.dump:
        run_case(args.case, None, args)
        return
    if args.case in ("all", "prefill"):
        lengths = [args.prefill_kv] if args.prefill_kv else range(16384, 131073, 16384)
        for length in lengths:
            run_case("prefill", length, args)
    if args.case in ("all", "decode"):
        run_case("decode", args.decode_kv, args)


if __name__ == "__main__":
    main()
