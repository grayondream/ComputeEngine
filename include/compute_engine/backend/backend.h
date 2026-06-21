#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/dtype.h"
#include "compute_engine/core/shape.h"
#include "compute_engine/core/tensor.h"
#include "compute_engine/core/device_type.h"
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace compute_engine {

struct BufferArg {
    enum Direction { INPUT, OUTPUT };
    TensorPtr tensor;
    Direction dir;

    static BufferArg Input(TensorPtr t) { return {std::move(t), INPUT}; }
    static BufferArg Output(TensorPtr t) { return {std::move(t), OUTPUT}; }
};

struct LaunchConfig {
    std::vector<int32_t> grid = {1, 1, 1};
    std::vector<int32_t> block = {1, 1, 1};
};

using KernelHandle = uint64_t;
using GraphHandle = void*;

class Device {
public:
    virtual ~Device() = default;
    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual Status Initialize() = 0;
    virtual Status Finalize() = 0;
    virtual bool IsAvailable() const = 0;
    virtual std::vector<DeviceInfo> EnumDevices() const = 0;
    virtual size_t GetMemoryUsage() const = 0;
    virtual size_t GetMaxMemory() const = 0;
    virtual bool HasTensorCore() const = 0;
};

class Allocator {
public:
    enum class CopyDirection { H2D, D2H, D2D };
    virtual ~Allocator() = default;
    virtual Status Allocate(void** ptr, size_t size) = 0;
    virtual Status Deallocate(void* ptr) = 0;
    virtual Status Copy(void* dst, const void* src,
                        size_t size, CopyDirection dir) = 0;
    virtual Status Memset(void* ptr, int value, size_t size) = 0;
    virtual Status CreateTensor(TensorPtr& tensor,
                                const Shape& shape, DType dtype) = 0;
    virtual size_t GetMemoryUsage() const { return 0; }
    virtual size_t GetMaxMemory() const { return 0; }
};

using AllocatorPtr = std::shared_ptr<Allocator>;

class StreamManager {
public:
    virtual ~StreamManager() = default;
    virtual int CreateStream() = 0;
    virtual Status DestroyStream(int stream_id) = 0;
    virtual Status SynchronizeStream(int stream_id) = 0;
    virtual Status SynchronizeAll() = 0;
    virtual Status RecordEvent(int stream_id, int event_id) = 0;
    virtual Status WaitEvent(int stream_id, int event_id) = 0;
    virtual Status QueryEvent(int event_id, bool* completed) = 0;
};

class KernelExecutor {
public:
    virtual ~KernelExecutor() = default;
    virtual Status CompileKernel(
        const std::string& name, const std::vector<uint8_t>& source,
        KernelHandle* handle) = 0;
    virtual Status LaunchKernel(
        KernelHandle handle, const std::vector<BufferArg>& args,
        const LaunchConfig& config, int stream_id = 0) = 0;
    virtual Status ReleaseKernel(KernelHandle handle) = 0;
    virtual Status BeginGraphCapture(int stream_id) = 0;
    virtual Status EndGraphCapture(GraphHandle* graph) = 0;
    virtual Status LaunchGraph(GraphHandle graph) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual Device*         device()    = 0;
    virtual Allocator*      allocator() = 0;
    virtual StreamManager*  stream()    = 0;
    virtual KernelExecutor* executor()  = 0;

    virtual Status Initialize() { return device()->Initialize(); }
    virtual Status Finalize() { return device()->Finalize(); }
    bool IsAvailable() const { return const_cast<Backend*>(this)->device()->IsAvailable(); }
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace compute_engine
