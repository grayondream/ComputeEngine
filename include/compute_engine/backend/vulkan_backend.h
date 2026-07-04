#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <vulkan/vulkan.h>

namespace compute_engine {

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    uint32_t queue_family_index = UINT32_MAX;
    uint32_t compute_queue_family = UINT32_MAX;
    char device_name[256]{};
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    VkPhysicalDeviceProperties props{};
    size_t total_global_mem = 0;
    bool initialized = false;

    static VulkanContext& Instance();
    Status Init();
    void Destroy();
    std::vector<DeviceInfo> EnumerateDevices() const;
};

class VulkanDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::VULKAN; }
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

class VulkanAllocator : public Allocator {
public:
    Status Allocate(void** ptr, size_t size) override;
    Status Deallocate(void* ptr) override;
    Status Copy(void* dst, const void* src,
                size_t size, CopyDirection dir) override;
    Status Memset(void* ptr, int value, size_t size) override;
    Status CreateTensor(TensorPtr& tensor,
                        const Shape& shape, DType dtype) override;
};

class VulkanStreamManager : public StreamManager {
public:
    VulkanStreamManager();
    ~VulkanStreamManager() override;

    int CreateStream() override;
    Status DestroyStream(int stream_id) override;
    Status SynchronizeStream(int stream_id) override;
    Status SynchronizeAll() override;
    Status RecordEvent(int stream_id, int event_id) override;
    Status WaitEvent(int stream_id, int event_id) override;
    Status QueryEvent(int event_id, bool* completed) override;

private:
    std::vector<VkCommandBuffer> cmd_buffers_;
    std::vector<VkFence> fences_;
    std::mutex mutex_;
};

struct VulkanKernelEntry {
    VkShaderModule shader_module;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    std::string name;
};

class VulkanKernelExecutor : public KernelExecutor {
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
    std::vector<VulkanKernelEntry> kernels_;
    std::mutex mutex_;
};

class VulkanBackend : public Backend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    Device*         device()    override { return &device_; }
    Allocator*      allocator() override { return &allocator_; }
    StreamManager*  stream()    override { return &stream_; }
    KernelExecutor* executor()  override { return &executor_; }

private:
    VulkanDevice device_;
    VulkanAllocator allocator_;
    VulkanStreamManager stream_;
    VulkanKernelExecutor executor_;
};

} // namespace compute_engine
