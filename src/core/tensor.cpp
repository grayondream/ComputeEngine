#include "compute_engine/core/tensor.h"
#include "compute_engine/memory/allocator.h"
#include <cstring>
#include <algorithm>
#include <sstream>
#include "compute_engine/platform/logger.h"

namespace compute_engine {

Storage::~Storage() {
    if (ptr_) {
        std::free(ptr_);
    }
}

Storage::Storage(Storage&& o) noexcept
    : device_type_(o.device_type_),
      ptr_(o.ptr_),
      byte_size_(o.byte_size_) {
    o.ptr_ = nullptr;
    o.byte_size_ = 0;
}

Storage& Storage::operator=(Storage&& o) noexcept {
    if (this != &o) {
        device_type_ = o.device_type_;
        ptr_ = o.ptr_;
        byte_size_ = o.byte_size_;
        o.ptr_ = nullptr;
        o.byte_size_ = 0;
    }
    return *this;
}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(std::move(shape)), dtype_(dtype) {
    ComputeStrides();
    size_t sz = byte_size();
    if (sz > 0) {
        void* ptr = std::malloc(sz);
        std::memset(ptr, 0, sz);
        storage_ = std::make_shared<Storage>(
            DeviceType::CPU, ptr, sz);
    }
    LOGI("Tensor created shape={} dtype={}", shape_.ToString(), DTypeToString(dtype_));
}

Tensor::Tensor(Shape shape, DType dtype, StoragePtr storage)
    : shape_(std::move(shape)), dtype_(dtype), storage_(std::move(storage)) {
    ComputeStrides();
    LOGI("Tensor created shape={} dtype={}", shape_.ToString(), DTypeToString(dtype_));
}

Tensor::Tensor(Shape shape, DType dtype, void* data)
    : shape_(std::move(shape)), dtype_(dtype) {
    ComputeStrides();
    storage_ = std::make_shared<Storage>(
        DeviceType::CPU, data, byte_size());
}

void Tensor::ComputeStrides() {
    strides_.resize(shape_.ndim());
    if (shape_.ndim() == 0) return;
    strides_.back() = SizeOf(dtype_);
    for (int64_t i = static_cast<int64_t>(shape_.ndim()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

bool Tensor::contiguous() const {
    if (shape_.ndim() <= 1) return true;
    int64_t expected = SizeOf(dtype_);
    for (int64_t i = static_cast<int64_t>(shape_.ndim()) - 1; i >= 0; --i) {
        if (strides_[i] != expected) return false;
        expected = strides_[i] * shape_[i];
    }
    return true;
}

Status Tensor::Reshape(const Shape& new_shape) {
    if (new_shape.num_elements() != num_elements()) {
        return Status::InvalidArgument("Reshape: element count mismatch");
    }
    shape_ = new_shape;
    ComputeStrides();
    return Status::OK();
}

Tensor Tensor::ToContiguous() const {
    if (contiguous()) return *this;
    Tensor result(shape_, dtype_);
    std::memcpy(result.data(), data(), byte_size());
    return result;
}

Tensor Tensor::Clone() const {
    Tensor result(shape_, dtype_);
    if (data() && result.data()) {
        std::memcpy(result.data(), data(), byte_size());
    }
    return result;
}

Tensor Tensor::View(int64_t off, const Shape& new_shape) const {
    Tensor result(new_shape, dtype_, storage_);
    result.offset_ = offset_ + off;
    result.ComputeStrides();
    return result;
}

Tensor Tensor::Slice(size_t dim, int64_t begin, int64_t end) const {
    Shape new_shape = shape_;
    new_shape[dim] = end - begin;
    int64_t byte_offset = begin * strides_[dim];
    Tensor result(new_shape, dtype_, storage_);
    result.offset_ = offset_ + byte_offset;
    result.strides_ = strides_;
    return result;
}

Tensor Tensor::Transpose(const std::vector<int64_t>& perm) const {
    Tensor result;
    result.shape_ = Shape(std::vector<int64_t>(perm.size(), 0));
    result.strides_.resize(perm.size());
    result.dtype_ = dtype_;
    result.storage_ = storage_;
    result.offset_ = offset_;
    for (size_t i = 0; i < perm.size(); ++i) {
        result.shape_[i] = shape_[perm[i]];
        result.strides_[i] = strides_[perm[i]];
    }
    return result;
}

void Tensor::Swap(Tensor& other) {
    std::swap(shape_, other.shape_);
    std::swap(strides_, other.strides_);
    std::swap(dtype_, other.dtype_);
    std::swap(offset_, other.offset_);
    std::swap(storage_, other.storage_);
}

Tensor Tensor::Zeros(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (t.data()) std::memset(t.data(), 0, t.byte_size());
    return t;
}

Tensor Tensor::Ones(const Shape& shape, DType dtype) {
    Tensor t = Zeros(shape, dtype);
    if (dtype == DType::FLOAT32) {
        auto* p = t.data_as<float>();
        for (int64_t i = 0; i < t.num_elements(); ++i) p[i] = 1.0f;
    }
    return t;
}

std::string Tensor::ToString() const {
    std::ostringstream ss;
    ss << "Tensor(" << shape_.ToString() << ", " << DTypeToString(dtype_) << ")";
    return ss.str();
}

} // namespace compute_engine
