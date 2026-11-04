#pragma once

#include <string>

#include "core/base/error.h"
#include "core/config/feature_flags.h"
#include "core/ir/model_kind.h"
#include "hybrid/hybrid_config.h"
#include "validation/config.h"

namespace ie {

enum class GpuBackend { NONE, CUDA, VULKAN };

// Feature-gating precedence (P0.5 controller ruling):
//   * `features` (FeatureFlags) is the single authoritative global gate for
//     every optional capability. Nothing is enabled unless `features` allows
//     it.
//   * The legacy bools below (`enable_training`, `enable_validation`,
//     `enable_deterministic`) and the `hardware_adapt` / `validation` toggles
//     are per-engine narrowing mirror/overrides only. They may further
//     restrict a capability for a specific EngineConfig instance, but they
//     must never widen something that `features` has disabled.
//   * Default construction keeps every mirror equal to its corresponding
//     FeatureFlags default; see the contract test
//     `EngineConfigFeatureTest.DefaultMirrorsMatchFeatureFlags`.
struct EngineConfig {
    GpuBackend gpu_backend{GpuBackend::NONE};
    bool enable_cpu{true};
    bool enable_training{false};
    bool enable_validation{false};
    bool enable_deterministic{false};

    ModelKind default_model_kind{ModelKind::UNKNOWN};
    FeatureFlags features;

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
