#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace compute_engine {

struct OpenCLContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    char device_name[256]{};
    char platform_name[128]{};
    cl_ulong total_global_mem = 0;
    cl_uint max_compute_units = 0;
    char device_version[64]{};
    bool initialized = false;

    static OpenCLContext& Instance();
    Status Init();
    void Destroy();
    std::vector<DeviceInfo> EnumerateDevices() const;
};

class OpenCLDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::OPENCL; }
    std::string name() const override;
    std::string version() const override;
    Status Initialize() override;
    Status Finalize() override;
    bool IsAvailable() const override;
    std::vector<DeviceInfo> EnumDevices() const override;
    size_t GetMemoryUsage() const override;
    size_t GetMaxMemory() const override;
    bool HasTensorCore() const override { return false; }
};

class OpenCLAllocator : public Allocator {
public:
    Status Allocate(void** ptr, size_t size) override;
    Status Deallocate(void* ptr) override;
    Status Copy(void* dst, const void* src,
                size_t size, CopyDirection dir) override;
    Status Memset(void* ptr, int value, size_t size) override;
    Status CreateTensor(TensorPtr& tensor,
                        const Shape& shape, DType dtype) override;
};

class OpenCLStreamManager : public StreamManager {
public:
    OpenCLStreamManager();
    ~OpenCLStreamManager() override;

    int CreateStream() override;
    Status DestroyStream(int stream_id) override;
    Status SynchronizeStream(int stream_id) override;
    Status SynchronizeAll() override;
    Status RecordEvent(int stream_id, int event_id) override;
    Status WaitEvent(int stream_id, int event_id) override;
    Status QueryEvent(int event_id, bool* completed) override;

private:
    std::vector<cl_command_queue> queues_;
    std::vector<cl_event> events_;
    std::mutex mutex_;
};

struct OpenCLKernelEntry {
    cl_program program;
    cl_kernel kernel;
    std::string name;
};

class OpenCLKernelExecutor : public KernelExecutor {
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
    std::vector<OpenCLKernelEntry> kernels_;
    std::mutex mutex_;
};

class OpenCLBackend : public Backend {
public:
    OpenCLBackend();
    ~OpenCLBackend() override;

    Device*         device()    override { return &device_; }
    Allocator*      allocator() override { return &allocator_; }
    StreamManager*  stream()    override { return &stream_; }
    KernelExecutor* executor()  override { return &executor_; }

private:
    OpenCLDevice device_;
    OpenCLAllocator allocator_;
    OpenCLStreamManager stream_;
    OpenCLKernelExecutor executor_;
};

} // namespace compute_engine
