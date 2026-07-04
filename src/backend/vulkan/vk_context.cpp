#include "compute_engine/backend/vulkan_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstring>
#include <vector>

namespace compute_engine {

VulkanContext& VulkanContext::Instance() {
    static VulkanContext ctx;
    return ctx;
}

Status VulkanContext::Init() {
    if (initialized) return Status::OK();

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ComputeEngine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "ComputeEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    const char* layers[] = {
        "VK_LAYER_KHRONOS_validation"
    };

    const char* extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    uint32_t layer_count = 0;
#ifndef NDEBUG
    layer_count = 1;
#endif

    VkInstanceCreateInfo inst_info{};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = layer_count;
    inst_info.ppEnabledLayerNames = layer_count > 0 ? layers : nullptr;
    inst_info.enabledExtensionCount = 1;
    inst_info.ppEnabledExtensionNames = extensions;

    VkResult res = vkCreateInstance(&inst_info, nullptr, &instance);
    if (res != VK_SUCCESS) {
        LOGE("vkCreateInstance failed: {}", static_cast<int>(res));
        return Status::RuntimeError("vkCreateInstance failed");
    }

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    if (gpu_count == 0) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        LOGE("No Vulkan physical devices found");
        return Status::RuntimeError("No Vulkan physical devices");
    }

    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());

    physical_device = gpus[0];
    for (auto gpu : gpus) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(gpu, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device = gpu;
            break;
        }
    }

    vkGetPhysicalDeviceProperties(physical_device, &props);
    api_version = props.apiVersion;
    driver_version = props.driverVersion;
    std::strncpy(device_name, props.deviceName, sizeof(device_name) - 1);
    total_global_mem = 0;
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_global_mem = std::max<size_t>(
                total_global_mem,
                static_cast<size_t>(mem_props.memoryHeaps[i].size));
        }
    }

    uint32_t qfam_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qfam_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfams(qfam_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qfam_count, qfams.data());

    compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qfam_count; i++) {
        if (qfams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    if (compute_queue_family == UINT32_MAX) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        LOGE("No compute queue family found");
        return Status::RuntimeError("No compute queue family");
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qinfo{};
    qinfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qinfo.queueFamilyIndex = compute_queue_family;
    qinfo.queueCount = 1;
    qinfo.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures features{};

    const char* dev_exts[] = {
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME
    };

    VkDeviceCreateInfo dev_info{};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &qinfo;
    dev_info.pEnabledFeatures = &features;
    dev_info.enabledExtensionCount = 1;
    dev_info.ppEnabledExtensionNames = dev_exts;

    res = vkCreateDevice(physical_device, &dev_info, nullptr, &device);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        LOGE("vkCreateDevice failed: {}", static_cast<int>(res));
        return Status::RuntimeError("vkCreateDevice failed");
    }

    vkGetDeviceQueue(device, compute_queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    res = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
    if (res != VK_SUCCESS) {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        LOGE("vkCreateCommandPool failed: {}", static_cast<int>(res));
        return Status::RuntimeError("vkCreateCommandPool failed");
    }

    initialized = true;
    LOGI("Vulkan context initialized: device={} mem={} MB api={}.{}.{}",
         device_name, total_global_mem / (1024 * 1024),
         VK_VERSION_MAJOR(api_version), VK_VERSION_MINOR(api_version),
         VK_VERSION_PATCH(api_version));
    return Status::OK();
}

void VulkanContext::Destroy() {
    if (!initialized) return;
    if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    physical_device = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    command_pool = VK_NULL_HANDLE;
    initialized = false;
    LOGI("Vulkan context destroyed");
}

std::vector<DeviceInfo> VulkanContext::EnumerateDevices() const {
    std::vector<DeviceInfo> devs;
    if (!initialized) return devs;
    DeviceInfo info;
    info.name = device_name;
    info.type = DeviceType::VULKAN;
    info.memory_total = total_global_mem;
    info.version = std::to_string(VK_VERSION_MAJOR(api_version)) + "." +
                   std::to_string(VK_VERSION_MINOR(api_version)) + "." +
                   std::to_string(VK_VERSION_PATCH(api_version));
    info.has_tensor_core = false;
    devs.push_back(info);
    return devs;
}

} // namespace compute_engine
