#include "compute_engine/backend/cuda_backend.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_LOG(fmt, ...) fprintf(stdout, "[CUDA] " fmt "\n", ##__VA_ARGS__)
#define CUDA_ERR(fmt, ...) fprintf(stderr, "[CUDA] " fmt "\n", ##__VA_ARGS__)

namespace compute_engine {

CUDAContext& CUDAContext::Instance() {
    static CUDAContext ctx;
    return ctx;
}

Status CUDAContext::Init(int device_id) {
    if (initialized && device_id == this->device_id) {
        return Status::OK();
    }
    if (initialized) {
        Destroy();
    }

    auto cerr = cuInit(0);
    if (cerr != CUDA_SUCCESS) {
        CUDA_ERR("cuInit failed (error=%d)", cerr);
        return Status::RuntimeError("CUDA: cuInit failed");
    }

    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        CUDA_ERR("no devices found (%s)", cudaGetErrorString(err));
        return Status::RuntimeError("CUDA: no devices found");
    }
    if (device_id >= count) {
        CUDA_ERR("invalid device id %d", device_id);
        return Status::InvalidArgument("CUDA: invalid device id");
    }

    this->device_id = device_id;
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        CUDA_ERR("failed to set device %d (%s)", device_id, cudaGetErrorString(err));
        return Status::RuntimeError("CUDA: failed to set device");
    }

    cudaDeviceProp p{};
    err = cudaGetDeviceProperties(&p, device_id);
    if (err != cudaSuccess) {
        CUDA_ERR("failed to get device properties (%s)", cudaGetErrorString(err));
        return Status::RuntimeError("CUDA: failed to get device properties");
    }

    std::strncpy(device_name, p.name, sizeof(device_name) - 1);
    sm_count = p.multiProcessorCount;
    cc_major = p.major;
    cc_minor = p.minor;
    total_global_mem = p.totalGlobalMem;

    initialized = true;
    CUDA_LOG("initialized: %s (SM %d.%d, %d SMs, %zu GB)",
             device_name, cc_major, cc_minor, sm_count,
             total_global_mem / (1024 * 1024 * 1024));
    return Status::OK();
}

void CUDAContext::Destroy() {
    if (!initialized) return;
    CUDA_LOG("context destroyed");
    initialized = false;
}

std::vector<DeviceInfo> CUDAContext::EnumerateDevices() const {
    std::vector<DeviceInfo> infos;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return infos;
    }
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        DeviceInfo info;
        info.name = p.name;
        info.version = std::to_string(p.major) + "." + std::to_string(p.minor);
        info.type = DeviceType::CUDA;
        info.memory_total = p.totalGlobalMem;
        info.memory_free = 0;
        info.has_tensor_core = (p.major >= 7);
        infos.push_back(std::move(info));
    }
    return infos;
}

} // namespace compute_engine
