#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/ir/model_kind.h"

namespace ie {

// Declares when a kernel variant is applicable (docs/readme.md §6.2). Matching
// is capability/threshold based, never architecture-name based.
struct KernelVariant {
    std::string name;
    ModelKind target_kind{ModelKind::UNKNOWN};

    struct Requirements {
        std::vector<std::string> required_caps;
        std::vector<std::string> forbidden_caps;
        float min_tensor_to_vector_ratio{0.0f};
        float max_tensor_to_vector_ratio{1e9f};
        std::size_t min_vram{0};
        float min_bandwidth{0.0f};
        std::size_t min_l3_cache{0};
    } req;

    struct CostModel {
        float estimated_tflops{0.0f};
        float estimated_bandwidth_gbps{0.0f};
        float estimated_latency_us{0.0f};
    } cost;
};

}  // namespace ie
