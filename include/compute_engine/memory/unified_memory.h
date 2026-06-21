#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/backend/backend.h"

namespace compute_engine {

class UnifiedMemoryManager {
public:
    UnifiedMemoryManager(Backend* primary, Backend* secondary = nullptr);
    ~UnifiedMemoryManager() = default;

    Status Allocate(void** ptr, size_t size, DeviceType device);
    Status Migrate(TensorPtr tensor, DeviceType target_device);
    Status SwapOut(TensorPtr tensor);
    Status SwapIn(TensorPtr tensor);

private:
    Backend* primary_backend_;
    Backend* secondary_backend_;
};

} // namespace compute_engine
