// Torch entry for package-specific, compiled vLLM-Ascend EXEC_NPU_CMD calls.
#include "call.h"
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
using namespace flashinfer_aclnn;
struct Binding {
    Submit submit;
    void* workspace;
    void* launch;
    std::atomic<int64_t> bytes{-1};
    Binding(Submit s, void* w, void* l) : submit(s), workspace(w), launch(l) {}
};
std::mutex mutex;
std::unordered_map<std::string, std::unique_ptr<Binding>> bindings;
thread_local std::string error_message;

Binding& lookup(const std::string& token) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = bindings.find(token);
    TORCH_CHECK(it != bindings.end(), "Unknown verified ACLNN binding");
    return *it->second;
}
void execute(const std::string& token, at::TensorList inputs, at::TensorList outputs,
             const c10::List<c10::IValue>& attributes, at::IntArrayRef widths) {
    auto& binding = lookup(token);
    // Output lists may be empty. A wrapped number or a host operand can precede the NPU input.
    auto device = [](at::TensorList tensors) -> c10::optional<c10::Device> {
        for (const auto& t : tensors) if (t.device().is_privateuseone()) return t.device();
        return c10::nullopt;
    };
    auto npu = device(outputs);
    if (!npu) npu = device(inputs);
    TORCH_CHECK(npu, "ACLNN requires an NPU operand");
    c10_npu::NPUGuard guard(*npu);
    binding.submit({inputs, outputs, attributes, widths, binding.workspace, binding.launch, binding.bytes});
}
std::vector<int64_t> storage_formats(at::TensorList tensors) {
    std::vector<int64_t> formats;
    for (const auto& t : tensors)
        formats.push_back(t.device().is_privateuseone() ? at_npu::native::get_npu_format(t) : -1);
    return formats;
}
} // namespace

extern "C" __attribute__((visibility("default"))) int fi_bind_package(
        const char* token, flashinfer_aclnn::Submit submit, void* workspace, void* launch) noexcept {
    try {
        TORCH_CHECK(token && submit && workspace && launch, "Incomplete verified ACLNN package");
        std::lock_guard<std::mutex> lock(mutex);
        auto& binding = bindings[token];
        if (!binding) binding = std::make_unique<Binding>(submit, workspace, launch);
        TORCH_CHECK(binding->submit == submit && binding->workspace == workspace && binding->launch == launch,
                    "Cannot replace an immutable ACLNN binding");
        return 0;
    } catch (const std::exception& e) { error_message = e.what(); return -1; }
}
extern "C" __attribute__((visibility("default"))) const char* fi_aclnn_last_error() noexcept {
    return error_message.c_str();
}
extern "C" __attribute__((visibility("default"))) int fi_aclnn_last_workspace_size(
        const char* token, uint64_t* size) noexcept {
    try {
        TORCH_CHECK(token && size, "Invalid ACLNN workspace observation");
        auto bytes = lookup(token).bytes.load();
        TORCH_CHECK(bytes >= 0, "ACLNN binding has no observed workspace query");
        *size = static_cast<uint64_t>(bytes);
        return 0;
    } catch (const std::exception& e) { error_message = e.what(); return -1; }
}

TORCH_LIBRARY(flashinfer_aclnn, m) {
    m.def("_execute(str token, Tensor[] inputs, Tensor(a!)[] outputs, Any[] attributes, int[] widths) -> ()");
    m.def("_storage_formats(Tensor[] tensors) -> int[]");
}
TORCH_LIBRARY_IMPL(flashinfer_aclnn, PrivateUse1, m) {
    m.impl("_execute", TORCH_FN(execute));
    m.impl("_storage_formats", TORCH_FN(storage_formats));
}
