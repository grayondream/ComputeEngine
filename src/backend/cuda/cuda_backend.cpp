#include "compute_engine/backend/cuda_backend.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_LOG(fmt, ...) fprintf(stdout, "[CUDA] " fmt "\n", ##__VA_ARGS__)
#define CUDA_ERR(fmt, ...) fprintf(stderr, "[CUDA] " fmt "\n", ##__VA_ARGS__)

namespace compute_engine {

// ─── CUDADevice ───

std::string CUDADevice::name() const {
    auto& ctx = CUDAContext::Instance();
    return ctx.initialized ? ctx.device_name : "CUDA";
}

std::string CUDADevice::version() const {
    auto& ctx = CUDAContext::Instance();
    if (!ctx.initialized) return "0.0";
    return std::to_string(ctx.cc_major) + "." + std::to_string(ctx.cc_minor);
}

Status CUDADevice::Initialize() {
    return CUDAContext::Instance().Init(0);
}

Status CUDADevice::Finalize() {
    CUDAContext::Instance().Destroy();
    return Status::OK();
}

bool CUDADevice::IsAvailable() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<DeviceInfo> CUDADevice::EnumDevices() const {
    return CUDAContext::Instance().EnumerateDevices();
}

size_t CUDADevice::GetMemoryUsage() const {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
        return total - free;
    }
    return 0;
}

size_t CUDADevice::GetMaxMemory() const {
    auto& ctx = CUDAContext::Instance();
    return ctx.initialized ? ctx.total_global_mem : 0;
}

bool CUDADevice::HasTensorCore() const {
    auto& ctx = CUDAContext::Instance();
    return ctx.initialized && ctx.cc_major >= 7;
}

// ─── CUDAStreamManager ───

CUDAStreamManager::CUDAStreamManager() {
    streams_.reserve(8);
    events_.reserve(64);
}

CUDAStreamManager::~CUDAStreamManager() {
    for (auto s : streams_) {
        if (s) cudaStreamDestroy(s);
    }
    for (auto e : events_) {
        if (e) cudaEventDestroy(e);
    }
}

int CUDAStreamManager::CreateStream() {
    std::lock_guard<std::mutex> lock(mutex_);
    cudaStream_t s;
    auto err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
        CUDA_ERR("stream create failed: %s", cudaGetErrorString(err));
        return -1;
    }
    int id = static_cast<int>(streams_.size());
    streams_.push_back(s);
    CUDA_LOG("stream %d created", id);
    return id;
}

Status CUDAStreamManager::DestroyStream(int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return Status::InvalidArgument("Invalid stream id");
    }
    auto err = cudaStreamDestroy(streams_[stream_id]);
    if (err != cudaSuccess) {
        return Status::RuntimeError("CUDA stream destroy failed");
    }
    streams_[stream_id] = nullptr;
    return Status::OK();
}

Status CUDAStreamManager::SynchronizeStream(int stream_id) {
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return Status::InvalidArgument("Invalid stream id");
    }
    auto err = cudaStreamSynchronize(streams_[stream_id]);
    if (err != cudaSuccess) {
        return Status::RuntimeError("CUDA stream sync failed");
    }
    return Status::OK();
}

Status CUDAStreamManager::SynchronizeAll() {
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return Status::RuntimeError("CUDA device sync failed");
    }
    return Status::OK();
}

Status CUDAStreamManager::RecordEvent(int stream_id, int event_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (event_id >= static_cast<int>(events_.size())) {
        cudaEvent_t e;
        auto err = cudaEventCreate(&e);
        if (err != cudaSuccess) {
            return Status::RuntimeError("CUDA event create failed");
        }
        events_.push_back(e);
    }
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return Status::InvalidArgument("Invalid stream id");
    }
    auto err = cudaEventRecord(events_[event_id], streams_[stream_id]);
    if (err != cudaSuccess) {
        return Status::RuntimeError("CUDA event record failed");
    }
    return Status::OK();
}

Status CUDAStreamManager::WaitEvent(int stream_id, int event_id) {
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return Status::InvalidArgument("Invalid stream id");
    }
    if (event_id < 0 || event_id >= static_cast<int>(events_.size())) {
        return Status::InvalidArgument("Invalid event id");
    }
    auto err = cudaStreamWaitEvent(streams_[stream_id], events_[event_id], 0);
    if (err != cudaSuccess) {
        return Status::RuntimeError("CUDA stream wait event failed");
    }
    return Status::OK();
}

Status CUDAStreamManager::QueryEvent(int event_id, bool* completed) {
    if (!completed) return Status::InvalidArgument("Null completed pointer");
    if (event_id < 0 || event_id >= static_cast<int>(events_.size())) {
        *completed = true;
        return Status::OK();
    }
    auto err = cudaEventQuery(events_[event_id]);
    *completed = (err == cudaSuccess);
    return Status::OK();
}

// ─── CUDAKernelExecutor ───

Status CUDAKernelExecutor::CompileKernel(
    const std::string& name, const std::vector<uint8_t>& source,
    KernelHandle* handle) {
    if (!handle) return Status::InvalidArgument("Null handle");

    CUmodule module;
    auto err = cuModuleLoadData(&module, source.data());
    if (err != CUDA_SUCCESS) {
        CUDA_ERR("module load failed for kernel '%s': error=%d", name.c_str(), err);
        return Status::RuntimeError("CUDA module load failed");
    }

    CUfunction func;
    err = cuModuleGetFunction(&func, module, name.c_str());
    if (err != CUDA_SUCCESS) {
        cuModuleUnload(module);
        CUDA_ERR("function '%s' not found in module", name.c_str());
        return Status::NotFound("CUDA function not found: " + name);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    KernelHandle id = kernels_.size() + 1;
    kernels_.push_back({module, func, name});
    *handle = id;
    CUDA_LOG("kernel '%s' compiled (handle=%llu)", name.c_str(), (unsigned long long)id);
    return Status::OK();
}

Status CUDAKernelExecutor::LaunchKernel(
    KernelHandle handle, const std::vector<BufferArg>& args,
    const LaunchConfig& config, int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t idx = static_cast<size_t>(handle - 1);
    if (idx >= kernels_.size()) {
        return Status::InvalidArgument("Invalid kernel handle");
    }

    auto& entry = kernels_[idx];
    CUstream stream = nullptr;

    std::vector<void*> kernel_args;
    kernel_args.reserve(args.size());
    for (auto& a : args) {
        if (a.tensor && a.tensor->data()) {
            kernel_args.push_back(a.tensor->data());
        } else {
            kernel_args.push_back(nullptr);
        }
    }

    auto err = cuLaunchKernel(
        entry.function,
        static_cast<unsigned int>(config.grid[0]),
        static_cast<unsigned int>(config.grid[1]),
        static_cast<unsigned int>(config.grid[2]),
        static_cast<unsigned int>(config.block[0]),
        static_cast<unsigned int>(config.block[1]),
        static_cast<unsigned int>(config.block[2]),
        0, stream,
        kernel_args.data(), nullptr);

    if (err != CUDA_SUCCESS) {
        CUDA_ERR("kernel '%s' launch failed: error=%d", entry.name.c_str(), err);
        return Status::RuntimeError("CUDA kernel launch failed");
    }
    return Status::OK();
}

Status CUDAKernelExecutor::ReleaseKernel(KernelHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t idx = static_cast<size_t>(handle - 1);
    if (idx >= kernels_.size()) {
        return Status::InvalidArgument("Invalid kernel handle");
    }
    auto& entry = kernels_[idx];
    if (entry.module) {
        cuModuleUnload(entry.module);
        entry.module = nullptr;
        entry.function = nullptr;
    }
    CUDA_LOG("kernel '%s' released", entry.name.c_str());
    return Status::OK();
}

Status CUDAKernelExecutor::BeginGraphCapture(int stream_id) {
    (void)stream_id;
    return Status::NotImplemented("CUDA graph capture not yet implemented");
}

Status CUDAKernelExecutor::EndGraphCapture(GraphHandle* graph) {
    if (graph) *graph = nullptr;
    return Status::NotImplemented("CUDA graph capture not yet implemented");
}

Status CUDAKernelExecutor::LaunchGraph(GraphHandle graph) {
    (void)graph;
    return Status::NotImplemented("CUDA graph capture not yet implemented");
}

// ─── CUDABackend ───

CUDABackend::CUDABackend() {
    CUDA_LOG("CUDABackend created");
}

CUDABackend::~CUDABackend() {
    CUDA_LOG("CUDABackend destroyed");
}

} // namespace compute_engine
