#include "compute_engine/backend/cuda_backend.h"
#include "compute_engine/core/tensor.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_LOG(fmt, ...) fprintf(stdout, "[CUDA] " fmt "\n", ##__VA_ARGS__)
#define CUDA_ERR(fmt, ...) fprintf(stderr, "[CUDA] " fmt "\n", ##__VA_ARGS__)

namespace compute_engine {

Status CUDAAllocator::Allocate(void** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return Status::OK();
    }
    auto err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        CUDA_ERR("malloc failed (%zu bytes): %s", size, cudaGetErrorString(err));
        return Status::RuntimeError("CUDA alloc failed");
    }
    return Status::OK();
}

Status CUDAAllocator::Deallocate(void* ptr) {
    if (!ptr) return Status::OK();
    auto err = cudaFree(ptr);
    if (err != cudaSuccess) {
        CUDA_ERR("free failed: %s", cudaGetErrorString(err));
        return Status::RuntimeError("CUDA free failed");
    }
    return Status::OK();
}

Status CUDAAllocator::Copy(void* dst, const void* src,
                           size_t size, CopyDirection dir) {
    if (size == 0) return Status::OK();
    cudaMemcpyKind kind;
    switch (dir) {
        case CopyDirection::H2D: kind = cudaMemcpyHostToDevice; break;
        case CopyDirection::D2H: kind = cudaMemcpyDeviceToHost; break;
        case CopyDirection::D2D: kind = cudaMemcpyDeviceToDevice; break;
        default: return Status::InvalidArgument("Invalid copy direction");
    }
    auto err = cudaMemcpy(dst, src, size, kind);
    if (err != cudaSuccess) {
        CUDA_ERR("memcpy failed (%zu bytes): %s", size, cudaGetErrorString(err));
        return Status::RuntimeError("CUDA memcpy failed");
    }
    return Status::OK();
}

Status CUDAAllocator::Memset(void* ptr, int value, size_t size) {
    if (!ptr || size == 0) return Status::OK();
    auto err = cudaMemset(ptr, value, size);
    if (err != cudaSuccess) {
        CUDA_ERR("memset failed: %s", cudaGetErrorString(err));
        return Status::RuntimeError("CUDA memset failed");
    }
    return Status::OK();
}

Status CUDAAllocator::CreateTensor(TensorPtr& tensor,
                                    const Shape& shape, DType dtype) {
    auto sz = static_cast<size_t>(shape.num_elements()) * SizeOf(dtype);
    if (sz == 0) {
        tensor = std::make_shared<Tensor>(shape, dtype);
        return Status::OK();
    }
    void* dev_ptr = nullptr;
    auto st = Allocate(&dev_ptr, sz);
    if (!st.ok()) return st;
    auto storage = std::make_shared<Storage>(DeviceType::CUDA, dev_ptr, sz);
    tensor = std::make_shared<Tensor>(shape, dtype, std::move(storage));
    return Status::OK();
}

} // namespace compute_engine
