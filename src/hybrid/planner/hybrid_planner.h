#pragma once

#include <cstddef>
#include <vector>

#include "core/backend/backend_caps.h"
#include "core/base/error.h"
#include "core/ir/graph.h"
#include "hybrid/hybrid_config.h"

namespace ie {

// Hybrid placement planner (docs/readme.md §7.3). P0.5 only freezes the
// contract; plan() reports Unsupported.
class HybridPlanner {
public:
    struct NodePlacement {
        NodeId node{kInvalidNode};
        BackendKind backend{};
        float estimated_cost{0.0f};
        std::size_t memory_footprint{0};
        bool needs_transfer{false};
    };

    Result<std::vector<NodePlacement>> plan(const Graph&, const HybridConfig&) {
        return Error::unsupported("hybrid planning not implemented in P0.5");
    }
};

// Concrete placement decisions consumed by HybridScheduler (docs/readme.md
// §7.4). Defined after HybridPlanner because it reuses NodePlacement.
struct ExecutionPlan {
    std::vector<HybridPlanner::NodePlacement> placements;
};

}  // namespace ie
