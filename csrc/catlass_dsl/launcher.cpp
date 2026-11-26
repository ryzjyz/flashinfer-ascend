#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <acl/acl_rt.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

void launch(const std::string& name, std::uintptr_t function, std::uint32_t blocks,
            std::string payload, std::vector<at::Tensor> tensors) {
    TORCH_CHECK(function != 0 && blocks > 0, "Invalid CATLASS DSL kernel or block count");
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == stream.device() && tensor.has_storage(),
                    "CATLASS DSL tensors must have storage on the launch device");
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    if (payload.empty()) {
        payload.resize(sizeof(std::uint64_t), '\0');
    }

    // Keep the packet and its tensor owners alive until CANN consumes the launch.
    const auto raw_stream = stream.stream(false);
    auto submit = [function, blocks, raw_stream, payload = std::move(payload),
                   tensors = std::move(tensors)]() mutable -> int {
        return aclrtLaunchKernelWithHostArgs(reinterpret_cast<aclrtFuncHandle>(function), blocks,
                                             raw_stream, nullptr, payload.data(), payload.size(),
                                             nullptr, 0);
    };
    at_npu::native::OpCommand::RunOpApiV2(name, submit);
}

}  // namespace

PYBIND11_MODULE(_flashinfer_catlass_dsl_launcher, module) {
    module.def("launch", &launch, pybind11::call_guard<pybind11::gil_scoped_release>());
}
