#include "core/backend/engine_config.h"

namespace ie {

Status validate(const EngineConfig& config) {
    if (!config.enable_cpu && config.gpu_backend == GpuBackend::NONE) {
        return Error::invalid_argument("no execution device enabled");
    }
    if (config.hardware_adapt.autotune_budget_ms <= 0) {
        return Error::invalid_argument("autotune_budget_ms must be positive");
    }
    return {};
}

}  // namespace ie
