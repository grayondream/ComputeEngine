#pragma once

#include <string>

#include "core/base/error.h"
#include "core/ir/model_kind.h"

namespace ie {

enum class GpuBackend { NONE, CUDA, VULKAN };

struct HybridConfig {
    bool enable_hybrid{false};
    int gpu_layers{0};
};

struct ValidationConfig {
    bool enable{false};
    std::string trace_dir;
};

struct EngineConfig {
    GpuBackend gpu_backend{GpuBackend::NONE};
    bool enable_cpu{true};
    bool enable_training{false};
    bool enable_validation{false};
    bool enable_deterministic{false};

    ModelKind default_model_kind{ModelKind::UNKNOWN};

    struct HardwareAdaptation {
        bool auto_detect{true};
        bool enable_runtime_autotuning{true};
        bool persist_tuning_cache{true};
        std::string tuning_cache_dir;
        bool enable_capability_driven_selection{true};
        int autotune_budget_ms{100};
    } hardware_adapt;

    HybridConfig hybrid;
    ValidationConfig validation;
};

Status validate(const EngineConfig& config);

}  // namespace ie
