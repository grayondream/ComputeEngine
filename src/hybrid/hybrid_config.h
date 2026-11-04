#pragma once

#include <cstddef>
#include <vector>

#include "core/backend/backend_caps.h"

namespace ie {

// Hybrid execution configuration (docs/readme.md §7.2). Everything is
// disabled/empty by default; real hybrid execution is not implemented in P0.5.
struct HybridConfig {
    bool enable_hybrid{false};
    int gpu_layers{0};

    enum class Strategy {
        AUTO,
        LAYER_WISE,
        EXPERT_WISE,
        NEURON_WISE,
        PIPELINE,
        SPECULATIVE,
        OP_LEVEL,
    };
    Strategy strategy{};

    struct DeviceConfig {
        BackendKind backend{};
        std::size_t memory_budget{0};
        float compute_share{0.0f};
    };
    std::vector<DeviceConfig> devices;

    struct LayerPlacement {
        int layer_start{0};
        int layer_end{0};
        BackendKind target{};
    };
    std::vector<LayerPlacement> manual_placement;

    struct ExpertConfig {
        int max_experts_on_gpu{0};
        float hotness_threshold{0.0f};
        bool enable_prefetch{false};
    } expert_cfg;

    struct KVCacheConfig {
        std::size_t gpu_kv_budget{0};
        bool offload_to_cpu{false};
        std::size_t cpu_kv_budget{0};
    } kv_cfg;

    struct PipelineConfig {
        bool enable_prefetch{false};
        int prefetch_depth{0};
    } pipeline_cfg;
};

}  // namespace ie
