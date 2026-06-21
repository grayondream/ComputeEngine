#include "compute_engine/memory/unified_memory.h"
#include "compute_engine/platform/logger.h"

namespace compute_engine {

UnifiedMemoryManager::UnifiedMemoryManager(Backend* primary, Backend* secondary)
    : primary_backend_(primary), secondary_backend_(secondary) {
    LOGI("UnifiedMemoryManager created");
}

Status UnifiedMemoryManager::Allocate(void** ptr, size_t size, DeviceType device) {
    if (device == DeviceType::CPU) {
        return primary_backend_->allocator()->Allocate(ptr, size);
    }
    return primary_backend_->allocator()->Allocate(ptr, size);
}

Status UnifiedMemoryManager::Migrate(TensorPtr, DeviceType) {
    LOGW("Migrate not yet implemented");
    return Status::NotImplemented("Migrate not yet implemented");
}

Status UnifiedMemoryManager::SwapOut(TensorPtr) {
    LOGW("SwapOut not yet implemented");
    return Status::NotImplemented("SwapOut not yet implemented");
}

Status UnifiedMemoryManager::SwapIn(TensorPtr) {
    LOGW("SwapIn not yet implemented");
    return Status::NotImplemented("SwapIn not yet implemented");
}

} // namespace compute_engine
