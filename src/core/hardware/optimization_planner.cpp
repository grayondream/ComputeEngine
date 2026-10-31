#include "core/hardware/optimization_planner.h"

namespace ie {

OptimizationPlanner::Plan OptimizationPlanner::plan(
    const HardwareProfile& /*profile*/, const Graph& /*graph*/,
    ModelKind /*kind*/, const EngineConfig& /*config*/) {
    Plan plan;
    plan.preferred_kernels = {"cpu_scalar"};
    plan.reason = "p0-conservative";
    return plan;
}

}  // namespace ie
