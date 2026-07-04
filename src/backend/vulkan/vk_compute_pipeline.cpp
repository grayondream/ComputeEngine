#include "compute_engine/backend/vulkan_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstring>
#include <vector>

namespace compute_engine {

// ─── VulkanAllocator ───

static VkDeviceMemory allocate_vk_memory(VkDeviceSize size, uint32_t memory_type) {
    VulkanContext& ctx = VulkanContext::Instance();
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_type;

    VkDeviceMemory memory;
    if (vkAllocateMemory(ctx.device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return memory;
}

static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
    VulkanContext& ctx = VulkanContext::Instance();
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
};

Status VulkanAllocator::Allocate(void** ptr, size_t size) {
    if (size == 0) { *ptr = nullptr; return Status::OK(); }
    VulkanContext& ctx = VulkanContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("Vulkan not initialized");

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(ctx.device, &buf_info, nullptr, &buffer) != VK_SUCCESS) {
        return Status::RuntimeError("vkCreateBuffer failed");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &mem_reqs);

    uint32_t mem_type = find_memory_type(mem_reqs.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = allocate_vk_memory(mem_reqs.size, mem_type);
    if (memory == VK_NULL_HANDLE) {
        mem_type = find_memory_type(mem_reqs.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memory = allocate_vk_memory(mem_reqs.size, mem_type);
        if (memory == VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx.device, buffer, nullptr);
            return Status::RuntimeError("vkAllocateMemory failed");
        }
    }

    vkBindBufferMemory(ctx.device, buffer, memory, 0);

    auto* vkbuf = new VulkanBuffer{buffer, memory, size};
    *ptr = vkbuf;
    LOGI("VulkanAllocator::Allocate size={}", size);
    return Status::OK();
}

Status VulkanAllocator::Deallocate(void* ptr) {
    if (!ptr) return Status::OK();
    VulkanContext& ctx = VulkanContext::Instance();
    auto* vkbuf = static_cast<VulkanBuffer*>(ptr);
    vkDestroyBuffer(ctx.device, vkbuf->buffer, nullptr);
    vkFreeMemory(ctx.device, vkbuf->memory, nullptr);
    delete vkbuf;
    LOGI("VulkanAllocator::Deallocate");
    return Status::OK();
}

static VkCommandBuffer begin_single_time_cmd() {
    VulkanContext& ctx = VulkanContext::Instance();
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = ctx.command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd_buf;
    vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd_buf);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf, &begin_info);
    return cmd_buf;
}

static void end_single_time_cmd(VkCommandBuffer cmd_buf) {
    VulkanContext& ctx = VulkanContext::Instance();
    vkEndCommandBuffer(cmd_buf);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.queue);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd_buf);
}

Status VulkanAllocator::Copy(void* dst, const void* src,
                              size_t size, CopyDirection dir) {
    if (!dst || !src || size == 0) return Status::OK();
    VulkanContext& ctx = VulkanContext::Instance();

    auto* src_buf = static_cast<const VulkanBuffer*>(src);
    auto* dst_buf = static_cast<VulkanBuffer*>(dst);

    VkCommandBuffer cmd = begin_single_time_cmd();

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &copy_region);

    end_single_time_cmd(cmd);
    return Status::OK();
}

Status VulkanAllocator::Memset(void* ptr, int value, size_t size) {
    if (!ptr || size == 0) return Status::OK();
    VulkanContext& ctx = VulkanContext::Instance();

    VkDeviceMemory mem = static_cast<VulkanBuffer*>(ptr)->memory;
    void* mapped;
    if (vkMapMemory(ctx.device, mem, 0, size, 0, &mapped) == VK_SUCCESS) {
        std::memset(mapped, value, size);
        vkUnmapMemory(ctx.device, mem);
    }
    return Status::OK();
}

Status VulkanAllocator::CreateTensor(TensorPtr& tensor,
                                      const Shape& shape, DType dtype) {
    tensor = std::make_shared<Tensor>(shape, dtype);
    return Status::OK();
}

// ─── VulkanKernelExecutor ───

Status VulkanKernelExecutor::CompileKernel(
    const std::string& name, const std::vector<uint8_t>& source,
    KernelHandle* handle) {
    VulkanContext& ctx = VulkanContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("Vulkan not initialized");

    VkShaderModuleCreateInfo sm_info{};
    sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_info.codeSize = source.size();
    sm_info.pCode = reinterpret_cast<const uint32_t*>(source.data());

    VkShaderModule shader_module;
    if (vkCreateShaderModule(ctx.device, &sm_info, nullptr, &shader_module) != VK_SUCCESS) {
        LOGE("vkCreateShaderModule failed for kernel '{}'", name);
        return Status::RuntimeError("vkCreateShaderModule failed");
    }

    VkDescriptorSetLayoutBinding bindings[16];
    for (int i = 0; i < 16; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info{};
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 16;
    dsl_info.pBindings = bindings;

    VkDescriptorSetLayout ds_layout;
    if (vkCreateDescriptorSetLayout(ctx.device, &dsl_info, nullptr, &ds_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(ctx.device, shader_module, nullptr);
        return Status::RuntimeError("vkCreateDescriptorSetLayout failed");
    }

    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &ds_layout;

    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(ctx.device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx.device, ds_layout, nullptr);
        vkDestroyShaderModule(ctx.device, shader_module, nullptr);
        return Status::RuntimeError("vkCreatePipelineLayout failed");
    }

    VkComputePipelineCreateInfo cp_info{};
    cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_info.stage.module = shader_module;
    cp_info.stage.pName = "main";
    cp_info.layout = pipeline_layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cp_info,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, ds_layout, nullptr);
        vkDestroyShaderModule(ctx.device, shader_module, nullptr);
        return Status::RuntimeError("vkCreateComputePipelines failed");
    }

    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 16;

    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &pool_size;
    dp_info.maxSets = 16;

    VkDescriptorPool descriptor_pool;
    if (vkCreateDescriptorPool(ctx.device, &dp_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        vkDestroyPipeline(ctx.device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, ds_layout, nullptr);
        vkDestroyShaderModule(ctx.device, shader_module, nullptr);
        return Status::RuntimeError("vkCreateDescriptorPool failed");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    KernelHandle id = kernels_.size() + 1;
    kernels_.push_back({shader_module, pipeline_layout, pipeline,
                        ds_layout, descriptor_pool, name});
    if (handle) *handle = id;
    LOGI("Vulkan kernel compiled: '{}' handle={}", name, id);
    return Status::OK();
}

Status VulkanKernelExecutor::LaunchKernel(
    KernelHandle handle, const std::vector<BufferArg>& args,
    const LaunchConfig& config, int stream_id) {
    CE_UNUSED(stream_id);
    VulkanContext& ctx = VulkanContext::Instance();
    if (!ctx.initialized) return Status::RuntimeError("Vulkan not initialized");

    VulkanKernelEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t idx = handle - 1;
        if (idx >= kernels_.size()) {
            return Status::RuntimeError("Invalid kernel handle");
        }
        entry = &kernels_[idx];
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = ctx.command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd_buf;
    if (vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd_buf) != VK_SUCCESS) {
        return Status::RuntimeError("vkAllocateCommandBuffers failed");
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf, &begin_info);

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, entry->pipeline);

    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool = entry->descriptor_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts = &entry->descriptor_set_layout;

    VkDescriptorSet descriptor_set;
    vkAllocateDescriptorSets(ctx.device, &ds_alloc, &descriptor_set);

    std::vector<VkDescriptorBufferInfo> buf_infos(args.size());
    std::vector<VkWriteDescriptorSet> writes(args.size());
    for (size_t i = 0; i < args.size(); i++) {
        auto* vkbuf = static_cast<const VulkanBuffer*>(args[i].tensor->data());
        buf_infos[i].buffer = vkbuf->buffer;
        buf_infos[i].offset = 0;
        buf_infos[i].range = vkbuf->size;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }

    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            entry->pipeline_layout, 0, 1,
                            &descriptor_set, 0, nullptr);

    vkCmdDispatch(cmd_buf,
                  static_cast<uint32_t>(config.grid[0]),
                  static_cast<uint32_t>(config.grid[1]),
                  static_cast<uint32_t>(config.grid[2]));

    vkEndCommandBuffer(cmd_buf);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    if (vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd_buf);
        return Status::RuntimeError("vkQueueSubmit failed");
    }

    vkQueueWaitIdle(ctx.queue);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd_buf);

    LOGI("Vulkan kernel launched: '{}' grid={}x{}x{}",
         entry->name, config.grid[0], config.grid[1], config.grid[2]);
    return Status::OK();
}

Status VulkanKernelExecutor::ReleaseKernel(KernelHandle handle) {
    VulkanContext& ctx = VulkanContext::Instance();
    std::lock_guard<std::mutex> lock(mutex_);
    size_t idx = handle - 1;
    if (idx >= kernels_.size()) return Status::OK();
    auto& k = kernels_[idx];
    vkDestroyDescriptorPool(ctx.device, k.descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, k.descriptor_set_layout, nullptr);
    vkDestroyPipeline(ctx.device, k.pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, k.pipeline_layout, nullptr);
    vkDestroyShaderModule(ctx.device, k.shader_module, nullptr);
    LOGI("Vulkan kernel released: '{}'", k.name);
    return Status::OK();
}

Status VulkanKernelExecutor::BeginGraphCapture(int) {
    return Status::NotImplemented("Vulkan does not support graph capture");
}
Status VulkanKernelExecutor::EndGraphCapture(GraphHandle*) {
    return Status::NotImplemented("Vulkan does not support graph capture");
}
Status VulkanKernelExecutor::LaunchGraph(GraphHandle) {
    return Status::NotImplemented("Vulkan does not support graph capture");
}

} // namespace compute_engine
