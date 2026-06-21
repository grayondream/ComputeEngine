#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/dtype.h"
#include "compute_engine/core/shape.h"
#include "compute_engine/core/device_type.h"
#include <memory>
#include <vector>
#include <cstring>
#include <sstream>
#include <memory>

namespace compute_engine {

struct Storage {
    DeviceType device_type_ = DeviceType::CPU;
    void* ptr_ = nullptr;
    size_t byte_size_ = 0;

    Storage() = default;
    Storage(DeviceType dt, void* p, size_t sz)
        : device_type_(dt), ptr_(p), byte_size_(sz) {}
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&& o) noexcept;
    Storage& operator=(Storage&& o) noexcept;
};

using StoragePtr = std::shared_ptr<Storage>;

class Tensor {
public:
    Tensor() = default;

    Tensor(Shape shape, DType dtype);

    Tensor(Shape shape, DType dtype, StoragePtr storage);

    Tensor(Shape shape, DType dtype, void* data);

    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.ndim(); }
    int64_t num_elements() const { return shape_.num_elements(); }
    size_t byte_size() const { return num_elements() * SizeOf(dtype_); }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t offset() const { return offset_; }

    void* data() const { return storage_ ? storage_->ptr_ : nullptr; }

    template <typename T>
    T* data_as() const { return static_cast<T*>(data()); }

    template <typename T>
    T& item_at(size_t i) { return data_as<T>()[i]; }

    bool contiguous() const;

    Status Reshape(const Shape& new_shape);
    Tensor ToContiguous() const;
    Tensor Clone() const;

    Tensor View(int64_t offset, const Shape& shape) const;
    Tensor Slice(size_t dim, int64_t begin, int64_t end) const;
    Tensor Transpose(const std::vector<int64_t>& perm) const;

    void Swap(Tensor& other);

    static Tensor Zeros(const Shape& shape, DType dtype);
    static Tensor Ones(const Shape& shape, DType dtype);

    std::string ToString() const;

private:
    Shape shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::FLOAT32;
    int64_t offset_ = 0;
    StoragePtr storage_;

    void ComputeStrides();
};

using TensorPtr = std::shared_ptr<Tensor>;

inline TensorPtr MakeTensor(const Shape& shape, DType dtype) {
    return std::make_shared<Tensor>(shape, dtype);
}

} // namespace compute_engine
