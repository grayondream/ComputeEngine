#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/memory/allocator.h"
#include "core/memory/memory_kind.h"

namespace ie {

struct GradFn {
    virtual ~GradFn() = default;
};

class Tensor {
public:
    Tensor() = default;

    Shape shape;
    DType dtype{DType::F32};
    Device device{Device::CPU};
    Layout layout{Layout::ROW_MAJOR};
    BufferHandle buffer{nullptr};

    bool requires_grad{false};
    Tensor* grad{nullptr};

    bool is_leaf() const { return grad_fn_ == nullptr; }
    GradFn* grad_fn() const { return grad_fn_.get(); }

    int64_t numel() const { return shape.numel(); }
    std::size_t bytes() const {
        return static_cast<std::size_t>(numel()) * dtype_size(dtype);
    }

    template <class T>
    T* data() {
        return static_cast<T*>(buffer);
    }

    template <class T>
    const T* data() const {
        return static_cast<const T*>(buffer);
    }

    static Tensor from_data(std::initializer_list<float> data) {
        Tensor tensor;
        tensor.shape = Shape{{static_cast<int64_t>(data.size())}};
        tensor.dtype = DType::F32;
        if (data.size() == 0) {
            return tensor;
        }
        const std::size_t bytes = data.size() * sizeof(float);
        void* raw = std::malloc(bytes);
        if (raw == nullptr) {
            return tensor;
        }
        std::memcpy(raw, data.begin(), bytes);
        tensor.buffer = raw;
        tensor.owner_ = std::shared_ptr<void>(raw, [](void* ptr) {
            std::free(ptr);
        });
        return tensor;
    }

    // Allocated tensors keep the producing allocator alive through the shared
    // owner, so returned outputs may outlive Engine/Backend.
    static Result<Tensor> allocate(const Shape& shape, DType dtype,
                                   std::shared_ptr<Allocator> allocator,
                                   MemoryKind kind = MemoryKind::ACTIVATION) {
        if (!allocator) {
            return Error::invalid_argument("Tensor::allocate requires allocator");
        }
        if (!shape.is_concrete()) {
            return Error::invalid_argument("Tensor::allocate requires concrete shape");
        }
        const std::size_t bytes =
            static_cast<std::size_t>(shape.numel()) * dtype_size(dtype);
        auto handle = allocator->alloc(bytes, kind);
        if (!handle.has_value()) {
            return handle.error();
        }
        Tensor tensor;
        tensor.shape = shape;
        tensor.dtype = dtype;
        tensor.device = Device::CPU;
        tensor.layout = Layout::ROW_MAJOR;
        tensor.buffer = *handle;
        tensor.owner_ = std::shared_ptr<void>(
            *handle, [allocator](void* ptr) { allocator->free(ptr); });
        return tensor;
    }

private:
    std::shared_ptr<void> owner_;
    std::shared_ptr<GradFn> grad_fn_;
};

}  // namespace ie
