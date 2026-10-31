#pragma once

#include <string>
#include <vector>

#include "core/backend/engine_config.h"
#include "core/hardware/policies.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/ir/model_kind.h"
#include "core/memory/policy.h"

namespace ie {

class OptimizationPlanner {
public:
    struct Plan {
        std::vector<std::string> enabled_passes;
        std::vector<std::string> preferred_kernels;
        MemoryPolicy memory_policy;
        QuantPolicy quant_policy;
        std::string reason;
    };

    static Plan plan(const HardwareProfile& profile, const Graph& graph,
                     ModelKind kind, const EngineConfig& config);
};

}  // namespace ie
