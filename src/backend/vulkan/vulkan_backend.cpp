#include "compute_engine/backend/vulkan_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstring>
#include <vector>

namespace compute_engine {

// ─── VulkanDevice ───

std::string VulkanDevice::name() const {
    return VulkanContext::Instance().device_name;
}

std::string VulkanDevice::version() const {
    auto& ctx = VulkanContext::Instance();
    return std::to_string(VK_VERSION_MAJOR(ctx.api_version)) + "." +
           std::to_string(VK_VERSION_MINOR(ctx.api_version));
}

Status VulkanDevice::Initialize() {
    return VulkanContext::Instance().Init();
}

Status VulkanDevice::Finalize() {
    VulkanContext::Instance().Destroy();
    return Status::OK();
}

bool VulkanDevice::IsAvailable() const {
    uint32_t version = 0;
    VkResult res = vkEnumerateInstanceVersion(&version);
    return res == VK_SUCCESS;
}

std::vector<DeviceInfo> VulkanDevice::EnumDevices() const {
    return VulkanContext::Instance().EnumerateDevices();
}

size_t VulkanDevice::GetMemoryUsage() const {
    return 0;
}

size_t VulkanDevice::GetMaxMemory() const {
    return VulkanContext::Instance().total_global_mem;
}

// ─── VulkanStreamManager ───

VulkanStreamManager::VulkanStreamManager() {}
VulkanStreamManager::~VulkanStreamManager() {
    SynchronizeAll();
}

int VulkanStreamManager::CreateStream() {
    VulkanContext& ctx = VulkanContext::Instance();
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = ctx.command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd_buf;
    VkFence fence;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd_buf) != VK_SUCCESS) {
            LOGE("Vulkan: failed to allocate command buffer for stream");
            return -1;
        }
        VkFenceCreateInfo f_info{};
        f_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        f_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(ctx.device, &f_info, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd_buf);
            return -1;
        }
        int id = static_cast<int>(cmd_buffers_.size());
        cmd_buffers_.push_back(cmd_buf);
        fences_.push_back(fence);
        LOGI("Vulkan stream created: id={}", id);
        return id;
    }
}

Status VulkanStreamManager::DestroyStream(int stream_id) {
    VulkanContext& ctx = VulkanContext::Instance();
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_id >= 0 && stream_id < static_cast<int>(cmd_buffers_.size())) {
        vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd_buffers_[stream_id]);
        vkDestroyFence(ctx.device, fences_[stream_id], nullptr);
        cmd_buffers_[stream_id] = VK_NULL_HANDLE;
        fences_[stream_id] = VK_NULL_HANDLE;
    }
    return Status::OK();
}

Status VulkanStreamManager::SynchronizeStream(int stream_id) {
    VulkanContext& ctx = VulkanContext::Instance();
    if (stream_id >= 0 && stream_id < static_cast<int>(fences_.size())) {
        vkWaitForFences(ctx.device, 1, &fences_[stream_id], VK_TRUE, UINT64_MAX);
    }
    return Status::OK();
}

Status VulkanStreamManager::SynchronizeAll() {
    auto& ctx = VulkanContext::Instance();
    if (ctx.queue) vkQueueWaitIdle(ctx.queue);
    return Status::OK();
}

Status VulkanStreamManager::RecordEvent(int stream_id, int event_id) {
    CE_UNUSED(stream_id);
    CE_UNUSED(event_id);
    return Status::OK();
}

Status VulkanStreamManager::WaitEvent(int stream_id, int event_id) {
    CE_UNUSED(stream_id);
    CE_UNUSED(event_id);
    return Status::OK();
}

Status VulkanStreamManager::QueryEvent(int event_id, bool* completed) {
    if (completed) *completed = true;
    CE_UNUSED(event_id);
    return Status::OK();
}

// ─── VulkanBackend ───

VulkanBackend::VulkanBackend() {
    LOGI("VulkanBackend created");
}

VulkanBackend::~VulkanBackend() {
    LOGI("VulkanBackend destroyed");
}

} // namespace compute_engine
