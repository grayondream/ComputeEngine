#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace compute_engine {

struct CUDAContext {
    int device_id = 0;
    int sm_count = 0;
    int cc_major = 0;
    int cc_minor = 0;
    size_t total_global_mem = 0;
    char device_name[256]{};
    bool initialized = false;

    static CUDAContext& Instance();
    Status Init(int device_id = 0);
    void Destroy();
    std::vector<DeviceInfo> EnumerateDevices() const;
};

class CUDADevice : public Device {
public:
    DeviceType type() const override { return DeviceType::CUDA; }
    std::string name() const override;
    std::string version() const override;
    Status Initialize() override;
    Status Finalize() override;
    bool IsAvailable() const override;
    std::vector<DeviceInfo> EnumDevices() const override;
    size_t GetMemoryUsage() const override;
    size_t GetMaxMemory() const override;
    bool HasTensorCore() const override;
};

class CUDAAllocator : public Allocator {
public:
    Status Allocate(void** ptr, size_t size) override;
    Status Deallocate(void* ptr) override;
    Status Copy(void* dst, const void* src,
                size_t size, CopyDirection dir) override;
    Status Memset(void* ptr, int value, size_t size) override;
    Status CreateTensor(TensorPtr& tensor,
                        const Shape& shape, DType dtype) override;
};

class CUDAStreamManager : public StreamManager {
public:
    CUDAStreamManager();
    ~CUDAStreamManager() override;

    int CreateStream() override;
    Status DestroyStream(int stream_id) override;
    Status SynchronizeStream(int stream_id) override;
    Status SynchronizeAll() override;
    Status RecordEvent(int stream_id, int event_id) override;
    Status WaitEvent(int stream_id, int event_id) override;
    Status QueryEvent(int event_id, bool* completed) override;

private:
#ifdef __CUDACC__
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
#else
    std::vector<void*> streams_;
    std::vector<void*> events_;
#endif
    std::mutex mutex_;
};

class CUDAKernelExecutor : public KernelExecutor {
public:
    Status CompileKernel(
        const std::string& name, const std::vector<uint8_t>& source,
        KernelHandle* handle) override;
    Status LaunchKernel(
        KernelHandle handle, const std::vector<BufferArg>& args,
        const LaunchConfig& config, int stream_id = 0) override;
    Status ReleaseKernel(KernelHandle handle) override;
    Status BeginGraphCapture(int stream_id) override;
    Status EndGraphCapture(GraphHandle* graph) override;
    Status LaunchGraph(GraphHandle graph) override;

private:
    struct KernelEntry {
#ifdef __CUDACC__
        CUmodule module;
        CUfunction function;
#else
        void* module;
        void* function;
#endif
        std::string name;
    };
    std::vector<KernelEntry> kernels_;
    std::mutex mutex_;
};

class CUDABackend : public Backend {
public:
    CUDABackend();
    ~CUDABackend() override;

    Device*         device()    override { return &device_; }
    Allocator*      allocator() override { return &allocator_; }
    StreamManager*  stream()    override { return &stream_; }
    KernelExecutor* executor()  override { return &executor_; }

private:
    CUDADevice device_;
    CUDAAllocator allocator_;
    CUDAStreamManager stream_;
    CUDAKernelExecutor executor_;
};

} // namespace compute_engine
