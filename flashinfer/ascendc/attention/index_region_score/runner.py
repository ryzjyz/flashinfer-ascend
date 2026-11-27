"""Adapt IRS eager views to the generated ACLNN Tensor contract."""

from dataclasses import dataclass, replace

from flashinfer.ascendc._runtime import native_admission
from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    attributes = dict(signature.specializations)
    if len(signature.shapes) != 8:
        return ("index_region_score requires six inputs and two outputs",)
    shapes = list(signature.shapes)
    if attributes.get("attribute.return_score") == "False":
        if shapes[-2] != (0,):
            return ("IDs-only score must have shape (0,)",)
        shapes[-2] = (1,)
    elif len(shapes[-2]) == 3 and shapes[-2][-1] == 0:
        shapes[-2] = shapes[-2][:-1] + (1,)
    # The runner materializes ordinary eager views and exposes valid pitched
    # pages as a contiguous physical span before entering the native adapter.
    attributes["tensor_contiguity"] = "".join("x" if dtype == "none" else "1" for dtype in signature.dtypes)
    normalized = replace(signature, shapes=tuple(shapes),
                         physical_layouts=("ND",) * len(shapes),
                         specializations=tuple(attributes.items()))
    return native_admission(recipe, normalized)


def _dense(tensor, torch_npu):
    import torch

    if tensor is None:
        return None
    if torch_npu.get_npu_format(tensor) != 2:
        tensor = torch_npu.npu_format_cast(tensor, 2)
    if not tensor.is_contiguous() or tensor.data_ptr() % 32:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _paged_span(key, torch_npu):
    if torch_npu.get_npu_format(key) != 2 or key.dim() != 4:
        return None
    pages, summaries, _, dim = key.shape
    stride = key.stride()
    byte_size = key.element_size()
    span = (pages - 1) * stride[0] + summaries * dim
    if (stride[3] != 1 or summaries > 1 and stride[1] != dim or
            stride[0] < summaries * dim or stride[0] * byte_size % 32 or key.data_ptr() % 32 or
            (key.storage_offset() + span) * byte_size > key.untyped_storage().nbytes()):
        return None
    return key.as_strided((span,), (1,)), (pages, summaries, stride[0])


def _aligned_score_span(score, logical_regions, protected, torch_npu):
    import torch

    if score.dim() != 3 or score.dtype != torch.float32 or torch_npu.get_npu_format(score) != 2:
        return None
    groups, tokens, regions = score.shape
    pitch = (regions + 127) // 128 * 128
    strides = (tokens * pitch, pitch, 1)
    if (regions == 0 or regions != logical_regions or regions == pitch or
            tuple(score.stride()) != strides or score.data_ptr() % 32 or
            score.storage_offset() * score.element_size() % 32):
        return None
    elements = groups * tokens * pitch
    byte_size = score.element_size()
    if (score.storage_offset() + elements) * byte_size > score.untyped_storage().nbytes():
        return None
    begin, end = score.data_ptr(), score.data_ptr() + elements * byte_size
    for tensor in protected:
        if tensor is None or tensor.numel() == 0:
            continue
        # Include gaps in strided inputs so padding cannot alias an argument.
        span = 1 + sum((dimension - 1) * stride for dimension, stride in zip(tensor.shape, tensor.stride()))
        other_begin = tensor.data_ptr()
        other_end = other_begin + span * tensor.element_size()
        if max(begin, other_begin) < min(end, other_end):
            return None
    return score.as_strided((groups, tokens, pitch), strides)


@dataclass(frozen=True)
class IndexRegionScoreRunner:
    native: object

    def __call__(self, query, weights, key_summary, actual_kv_lengths, block_table,
                 cu_seq_lens, score, selected_region_ids, **attributes):
        import torch
        import torch_npu

        native_attributes = dict(attributes, graph_key_pages=0, graph_key_summaries=0, graph_key_page_stride=0)
        key = key_summary
        if attributes["cache_mode"] == 1 and not key.is_contiguous():
            span = _paged_span(key, torch_npu)
            if span is not None:
                key, metadata = span
                native_attributes.update(zip(("graph_key_pages", "graph_key_summaries", "graph_key_page_stride"), metadata))
        inputs = tuple(_dense(tensor, torch_npu) for tensor in (
            query, weights, key, actual_kv_lengths, block_table, cu_seq_lens))
        score_output = score
        aligned_score = None
        if not attributes["return_score"]:
            score_output = torch.empty((1,), dtype=torch.float32, device=query.device)
        elif score.numel() == 0:
            score_output = torch.empty(tuple(score.shape[:-1]) + (1,), dtype=torch.float32, device=query.device)
        elif not score.is_contiguous() or score.data_ptr() % 32 or torch_npu.get_npu_format(score) != 2:
            protected = (query, weights, key_summary, actual_kv_lengths, block_table,
                         cu_seq_lens, selected_region_ids) + inputs
            logical_regions = attributes["max_region_len"]
            if logical_regions == -1:
                logical_regions = key_summary.shape[0]
            aligned_score = _aligned_score_span(score, logical_regions, protected, torch_npu)
            score_output = (aligned_score if aligned_score is not None else
                            torch.empty(tuple(score.shape), dtype=score.dtype, device=score.device))
        ids_output = selected_region_ids
        if (not ids_output.is_contiguous() or ids_output.data_ptr() % 32 or
                torch_npu.get_npu_format(ids_output) != 2):
            ids_output = torch.empty(tuple(ids_output.shape), dtype=ids_output.dtype, device=ids_output.device)
        result = self.native(*inputs, score_output, ids_output, **native_attributes)
        if aligned_score is None and score_output is not score and attributes["return_score"] and score.numel():
            regions = attributes["max_region_len"]
            if regions == -1:
                regions = key_summary.shape[0]
            score[..., :regions].copy_(score_output[..., :regions])
        if ids_output is not selected_region_ids:
            selected_region_ids.copy_(ids_output)
        return result


def make_runner(module, spec, recipe):
    return IndexRegionScoreRunner(package_tensor_runner(module, recipe, spec=spec))
