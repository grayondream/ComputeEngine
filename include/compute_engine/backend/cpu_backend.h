#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include "compute_engine/memory/allocator.h"
#include <string>
#include <vector>
#include <memory>

namespace compute_engine {

class CPUDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::CPU; }
    std::string name() const override { return "CPU"; }
    std::string version() const override { return "1.0"; }
    Status Initialize() override;
    Status Finalize() override;
    bool IsAvailable() const override { return true; }
    std::vector<DeviceInfo> EnumDevices() const override;
    size_t GetMemoryUsage() const override;
    size_t GetMaxMemory() const override;
    bool HasTensorCore() const override { return false; }
};

class CPUStreamManager : public StreamManager {
public:
    int CreateStream() override;
    Status DestroyStream(int stream_id) override;
    Status SynchronizeStream(int stream_id) override;
    Status SynchronizeAll() override;
    Status RecordEvent(int stream_id, int event_id) override;
    Status WaitEvent(int stream_id, int event_id) override;
    Status QueryEvent(int event_id, bool* completed) override;
};

class CPUKernelExecutor : public KernelExecutor {
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
};

class CPUBackend : public Backend {
public:
    CPUBackend();
    ~CPUBackend() override;

    Device*         device()    override { return &device_; }
    Allocator*      allocator() override { return &allocator_; }
    StreamManager*  stream()    override { return &stream_; }
    KernelExecutor* executor()  override { return &executor_; }

private:
    CPUDevice device_;
    CPUAllocator allocator_;
    CPUStreamManager stream_;
    CPUKernelExecutor executor_;
};

} // namespace compute_engine
