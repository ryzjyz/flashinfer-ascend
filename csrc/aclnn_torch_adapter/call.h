// Interface between the common Torch dispatcher and a package's compiled EXEC_NPU_CMD call.
#pragma once
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <atomic>

namespace flashinfer_aclnn {
struct Call {
    at::TensorList inputs;
    at::TensorList outputs;
    const c10::List<c10::IValue>& attributes;
    at::IntArrayRef widths;
    void* workspace_fn;
    void* launch_fn;
    std::atomic<int64_t>& workspace_bytes;
};
using Submit = void (*)(const Call&);
} // namespace flashinfer_aclnn
