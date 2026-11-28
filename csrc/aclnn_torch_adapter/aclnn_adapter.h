// Included only by the per-package typed call site generated at build time.
#pragma once
#include <acl/acl_base.h>
#include "call.h"
#include <algorithm>
#include <deque>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#define FI_ACLNN_WORKSPACE_FN(api) invocation.workspace_fn
#define FI_ACLNN_LAUNCH_FN(api) invocation.launch_fn
#define FI_ACLNN_OBSERVE_WORKSPACE(n) invocation.workspace_bytes.store(static_cast<int64_t>(n))
#define FI_ACLNN_RETAIN_ARGUMENTS() operands.keepalive
#include "op_api_common.h"

namespace flashinfer_aclnn {
struct Owners {
    std::vector<at::Tensor> tensors;
    c10::List<c10::IValue> attributes;
    std::deque<std::vector<at::Tensor>> wrapped;
    explicit Owners(const Call& c) : attributes(c.attributes) {
        tensors.insert(tensors.end(), c.inputs.begin(), c.inputs.end());
        tensors.insert(tensors.end(), c.outputs.begin(), c.outputs.end());
    }
};

class Operands {
    const Call& call_;
    at::TensorList inputs_, outputs_;
    at::IntArrayRef widths_;
    std::vector<std::pair<size_t, size_t>> aliases_;
    const c10_npu::NPUStream stream_ = c10_npu::getCurrentNPUStream();

    static size_t flat_index(at::TensorList values, const at::Tensor* slot) {
        auto it = std::find_if(values.begin(), values.end(), [slot](const at::Tensor& t) { return &t == slot; });
        TORCH_INTERNAL_ASSERT(it != values.end(), "ACLNN alias must refer to a resolved operand");
        return it - values.begin();
    }

    static size_t strided_bytes(const at::Tensor& tensor, bool writable) {
        if (tensor.numel() == 0) return 0;
        const auto itemsize = tensor.element_size();
        const auto capacity = tensor.storage().nbytes() / itemsize;
        const auto offset = tensor.storage_offset();
        TORCH_CHECK(offset >= 0 && static_cast<size_t>(offset) < capacity,
                    "ACLNN strided Tensor exceeds its storage span");
        const size_t available = capacity - offset;
        size_t span = 1;
        c10::SmallVector<std::pair<size_t, size_t>, 5> axes;
        for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
            const auto stride = tensor.stride(dim);
            TORCH_CHECK(stride >= 0, "ACLNN strided Tensor requires nonnegative strides");
            if (tensor.size(dim) <= 1) continue;
            const auto steps = static_cast<size_t>(tensor.size(dim) - 1);
            TORCH_CHECK(stride > 0, "ACLNN strided Tensor requires positive strides on non-singleton axes");
            TORCH_CHECK(static_cast<size_t>(stride) <= (available - span) / steps,
                        "ACLNN strided Tensor exceeds its storage span");
            span += steps * stride;
            if (writable) axes.emplace_back(stride, steps);
        }
        if (writable) {
            std::sort(axes.begin(), axes.end());
            size_t occupied = 1;
            for (const auto& [stride, steps] : axes) {
                TORCH_CHECK(stride >= occupied, "ACLNN strided output has internally overlapping elements");
                occupied += steps * stride;
            }
        }
        return span * itemsize;
    }

    static std::pair<uintptr_t, uintptr_t> range(const at::Tensor& tensor) {
        if (tensor.numel() == 0) return {0, 0};
        if (at_npu::native::get_npu_format(tensor) == ACL_FORMAT_FRACTAL_NZ) {
            const auto begin = reinterpret_cast<uintptr_t>(tensor.storage().data());
            return {begin, begin + tensor.storage().nbytes()};
        }
        const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const auto bytes = tensor.is_contiguous() ? tensor.nbytes() : strided_bytes(tensor, false);
        return {begin, begin + bytes};
    }

public:
    const std::shared_ptr<Owners> keepalive;
    explicit Operands(const Call& c) : call_(c), inputs_(c.inputs), outputs_(c.outputs), widths_(c.widths),
            keepalive(std::make_shared<Owners>(c)) {}

    void exact_alias(at::TensorList input, at::TensorList output) {
        TORCH_INTERNAL_ASSERT(input.size() == 1 && output.size() == 1);
        const auto& a = input.front();
        const auto& b = output.front();
        TORCH_CHECK(a.device() == b.device() && a.is_alias_of(b) &&
                    a.scalar_type() == b.scalar_type() && a.storage_offset() == b.storage_offset() &&
                    a.sizes() == b.sizes() && a.strides() == b.strides(),
                    "ACLNN alias pair must be the exact same storage view");
        aliases_.emplace_back(flat_index(call_.inputs, input.data()), flat_index(call_.outputs, output.data()));
    }
    // -1 is an absent optional argument. Zero is a present empty TensorList.
    c10::optional<at::TensorList> take(bool output, bool list, bool optional, int layouts, bool cpu,
                                      bool allow_strided = false) {
        auto& source = output ? outputs_ : inputs_;
        int64_t width = 1;
        if (!call_.widths.empty()) {
            TORCH_CHECK(!widths_.empty(), "Missing ACLNN operand width");
            width = widths_.front();
            widths_ = widths_.slice(1);
        }
        if (width == -1 && optional) return c10::nullopt;
        TORCH_CHECK(width >= 0 && static_cast<size_t>(width) <= source.size() && (list || width == 1),
                    "Invalid ACLNN Tensor/List argument count");
        auto tensors = source.slice(0, width);
        source = source.slice(width);
        for (const auto& tensor : tensors) {
            if (cpu) {
                TORCH_CHECK(tensor.device().is_cpu(), "Expected a declared CPU tensor");
                continue;
            }
            TORCH_CHECK(tensor.device() == stream_.device(),
                        "ACLNN tensor violates its declared device/format");
            const auto format = at_npu::native::get_npu_format(tensor);
            const int mask = format == ACL_FORMAT_ND ? 1 : format == ACL_FORMAT_FRACTAL_NZ ? 2 : 0;
            TORCH_CHECK(layouts & mask, "ACLNN tensor violates its declared device/format");
            if (!tensor.is_contiguous()) {
                TORCH_CHECK(allow_strided && format == ACL_FORMAT_ND,
                            "ACLNN tensor violates its declared device/format");
                strided_bytes(tensor, output);
            }
            if (format == ACL_FORMAT_FRACTAL_NZ && tensor.dim() == 2 && tensor.scalar_type() == at::kChar) {
                // INT8 NZ stores N in groups of 32 and M in groups of 16.
                const std::vector<int64_t> expected = {
                    (tensor.size(1) + 31) / 32, (tensor.size(0) + 15) / 16, 16, 32};
                TORCH_CHECK(at_npu::native::get_npu_storage_sizes(tensor) == expected,
                            "ACLNN INT8 FRACTAL_NZ storage shape does not match its logical orientation");
            }
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream_);
        }
        // Wrapped CPU numbers use device storage. Retain the copies through queued submission.
        if (cpu && std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
                return t.unsafeGetTensorImpl()->is_wrapped_number();
            })) {
            auto& copied = keepalive->wrapped.emplace_back(tensors.begin(), tensors.end());
            for (auto& t : copied) {
                if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
                    t = CopyTensorHostToDevice(t);
                    c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream_);
                }
            }
            return at::TensorList(copied);
        }
        return tensors;
    }
    void finish() const {
        TORCH_CHECK(inputs_.empty() && outputs_.empty() && widths_.empty(), "Extra ACLNN operands");
        for (size_t out = 0; out < call_.outputs.size(); ++out) {
            const auto& target = call_.outputs[out];
            const auto a = range(target);
            auto overlaps = [&](const at::Tensor& other) {
                if (other.device() != target.device()) return false;
                const auto b = range(other);
                return std::max(a.first, b.first) < std::min(a.second, b.second);
            };
            for (size_t input = 0; input < call_.inputs.size(); ++input) {
                const auto pair = std::make_pair(input, out);
                TORCH_CHECK(std::find(aliases_.begin(), aliases_.end(), pair) != aliases_.end()
                                || !overlaps(call_.inputs[input]),
                            "ACLNN output tensors must not overlap other arguments");
            }
            for (size_t prior = 0; prior < out; ++prior) {
                TORCH_CHECK(!overlaps(call_.outputs[prior]), "ACLNN outputs must not overlap");
            }
        }
    }
};

// NativeBinding storage conversions change the descriptor, without copying the Tensor.
enum class StorageKind { Int4, UInt64, Float4E2M1, Float8E8M0 };
struct StorageView { at::Tensor tensor; StorageKind kind; };
inline aclTensor* ConvertType(const StorageView& v) {
    const auto& t = v.tensor;
    if (!t.defined()) return nullptr;
    const bool int4 = v.kind == StorageKind::Int4 && t.scalar_type() == at::kInt;
    const bool uint64 = v.kind == StorageKind::UInt64 && t.scalar_type() == at::kLong;
    const bool fp4 = v.kind == StorageKind::Float4E2M1 && t.scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    const bool e8m0 = v.kind == StorageKind::Float8E8M0 && t.scalar_type() == at::ScalarType::Float8_e8m0fnu;
    if (!int4 && !uint64 && !fp4 && !e8m0) return ::ConvertType(t);
    auto sizes = t.sizes().vec(), strides = t.strides().vec();
    auto offset = t.storage_offset();
    auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(t));
    auto storage = format == ACL_FORMAT_FRACTAL_NZ ? at_npu::native::get_npu_storage_sizes(t)
        : std::vector<int64_t>{static_cast<int64_t>(t.storage().nbytes() / t.element_size())};
    if (fp4 || e8m0) {
        TORCH_CHECK(format == ACL_FORMAT_ND && t.element_size() == 1 && !sizes.empty(),
                    "MXFP4/E8M0 require ND byte-addressable storage");
    }
    if (int4 || fp4) {
        TORCH_CHECK(!sizes.empty() && !storage.empty(), "Packed storage requires dimensional metadata");
        TORCH_CHECK(!fp4 || strides.back() == 1, "MXFP4 packed dimension must be contiguous");
        const int64_t multiplier = int4 ? 8 : 2;
        const auto unpack = [multiplier](int64_t value) {
            TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max() / multiplier,
                        "ACLNN packed shape, stride or offset overflows");
            return value * multiplier;
        };
        sizes.back() = unpack(sizes.back());
        for (size_t i = 1; i < strides.size(); ++i) strides[i - 1] = unpack(strides[i - 1]);
        storage.back() = unpack(storage.back());
        offset = unpack(offset);
    }
    if (int4 && format == ACL_FORMAT_FRACTAL_NZ) {
        TORCH_CHECK(sizes.size() >= 2 && sizes.back() % 64 == 0 && sizes[sizes.size()-2] % 16 == 0,
                    "Native INT4 NZ requires N aligned to 64 and K aligned to 16");
        storage.assign(sizes.begin(), sizes.end()-2);
        storage.insert(storage.end(), {sizes.back()/64, sizes[sizes.size()-2]/16, 16, 64});
    }
    static const auto create = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create, "aclCreateTensor unavailable");
    const auto dtype = int4 ? ACL_INT4 : uint64 ? ACL_UINT64 : fp4 ? ACL_FLOAT4_E2M1 : ACL_FLOAT8_E8M0;
    auto* descriptor = create(sizes.data(), sizes.size(), dtype,
                              strides.data(), offset, format, storage.data(), storage.size(),
                              const_cast<void*>(t.storage().data()));
    TORCH_CHECK(descriptor, "aclCreateTensor failed for packed storage");
    return descriptor;
}

template <typename T>
std::vector<T> array_values(const c10::IValue& value) {
    // A typed recipe determines the type of an empty sequence as well.
    const auto& elements = value.toTupleRef().elements();
    std::vector<T> data;
    data.reserve(elements.size());
    for (const auto& element : elements) data.push_back(element.to<T>());
    return data;
}
class BoolValues {
    std::unique_ptr<bool[]> ptr_;
    size_t size_;
public:
    explicit BoolValues(const c10::IValue& value) : size_(value.toTupleRef().elements().size()) {
        ptr_ = std::make_unique<bool[]>(size_);
        for (size_t i = 0; i < size_; ++i) ptr_[i] = value.toTupleRef().elements()[i].toBool();
    }
    at::ArrayRef<bool> view() const { return at::ArrayRef<bool>(ptr_.get(), size_); }
};
} // namespace flashinfer_aclnn
