#include "compute_engine/memory/allocator.h"
#include "compute_engine/platform/logger.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace compute_engine {

Status CPUAllocator::Allocate(void** ptr, size_t size) {
    if (size == 0) { *ptr = nullptr; return Status::OK(); }
    void* p = std::malloc(size + 64);
    if (!p) return Status::RuntimeError("malloc failed");
    *ptr = p;
    LOGI("CPUAllocator::Allocate size={}", size);
    return Status::OK();
}

Status CPUAllocator::Deallocate(void* ptr) {
    std::free(ptr);
    LOGI("CPUAllocator::Deallocate");
    return Status::OK();
}

Status CPUAllocator::Copy(void* dst, const void* src, size_t size, CopyDirection) {
    if (dst && src && size > 0) std::memcpy(dst, src, size);
    return Status::OK();
}

Status CPUAllocator::Memset(void* ptr, int value, size_t size) {
    if (ptr && size > 0) std::memset(ptr, value, size);
    return Status::OK();
}

Status CPUAllocator::CreateTensor(TensorPtr& tensor, const Shape& shape, DType dtype) {
    tensor = std::make_shared<Tensor>(shape, dtype);
    return Status::OK();
}

} // namespace compute_engine
