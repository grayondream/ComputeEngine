#include "compute_engine/backend/backend_registry.h"
#include "compute_engine/backend/cpu_backend.h"
#include "compute_engine/platform/logger.h"

#ifdef CE_HAS_CUDA
#include "compute_engine/backend/cuda_backend.h"
#endif
#ifdef CE_HAS_OPENCL
#include "compute_engine/backend/opencl_backend.h"
#endif
#ifdef CE_HAS_VULKAN
#include "compute_engine/backend/vulkan_backend.h"
#endif

namespace compute_engine {

void BackendRegistry::InitBuiltinBackends() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    auto& reg = Instance();
    reg.Register(DeviceType::CPU,
                 []() -> BackendPtr { return std::make_shared<CPUBackend>(); },
                 100);

#ifdef CE_HAS_CUDA
    reg.Register(DeviceType::CUDA,
                 []() -> BackendPtr { return std::make_shared<CUDABackend>(); },
                 200);
#endif

#ifdef CE_HAS_OPENCL
    reg.Register(DeviceType::OPENCL,
                 []() -> BackendPtr { return std::make_shared<OpenCLBackend>(); },
                 150);
#endif

#ifdef CE_HAS_VULKAN
    reg.Register(DeviceType::VULKAN,
                 []() -> BackendPtr { return std::make_shared<VulkanBackend>(); },
                 150);
#endif
}

BackendRegistry& BackendRegistry::Instance() {
    static BackendRegistry instance;
    LOGI("BackendRegistry initialized");
    return instance;
}

void BackendRegistry::Register(DeviceType type,
                                std::function<BackendPtr()> factory,
                                int priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& e : entries_) {
        if (e.type == type) {
            e.factory = std::move(factory);
            e.priority = priority;
            return;
        }
    }
    entries_.push_back({type, std::move(factory), priority});
    LOGI("Backend registered: type={} priority={}", DeviceTypeToString(type), priority);
}

BackendPtr BackendRegistry::Get(DeviceType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& e : entries_) {
        if (e.type == type) {
            auto b = e.factory();
            if (b && b->IsAvailable()) {
                LOGI("Backend found for type={}", DeviceTypeToString(type));
                return b;
            }
        }
    }
    LOGW("No available backend for type={}", DeviceTypeToString(type));
    return nullptr;
}

BackendPtr BackendRegistry::GetBestAvailable(
    const std::vector<DeviceType>& preferences) {
    for (auto type : preferences) {
        auto b = Get(type);
        if (b) return b;
    }
    return nullptr;
}

BackendPtr BackendRegistry::GetBestAvailable() {
    if (!priority_list_.empty()) {
        return GetBestAvailable(priority_list_);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Entry> sorted = entries_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Entry& a, const Entry& b) { return a.priority > b.priority; });
    for (auto& e : sorted) {
        auto b = e.factory();
        if (b && b->IsAvailable()) return b;
    }
    return nullptr;
}

std::vector<BackendPtr> BackendRegistry::GetAllAvailable() {
    LOGI("Getting all available backends");
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BackendPtr> result;
    for (auto& e : entries_) {
        auto b = e.factory();
        if (b && b->IsAvailable()) result.push_back(b);
    }
    return result;
}

void BackendRegistry::SetPriorityList(const std::vector<DeviceType>& ordered) {
    std::lock_guard<std::mutex> lock(mutex_);
    priority_list_ = ordered;
    LOGI("Backend priority list set ({} devices)", ordered.size());
}

} // namespace compute_engine
