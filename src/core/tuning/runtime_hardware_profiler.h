#pragma once

#include "core/hardware/optimization_planner.h"
#include "core/hardware/profile.h"
#include "core/ir/model_kind.h"

namespace ie {

// Layer-3 hardware profiling. Measurements are stored for forward
// compatibility, but replanning is a stub in P0.5.
class RuntimeHardwareProfiler {
public:
    struct MeasuredPerformance {
        float achieved_tflops{0.0f};
        float achieved_bandwidth_gbps{0.0f};
        float achieved_latency_us{0.0f};
        float power_draw_watts{0.0f};
        float temperature_c{0.0f};
        float clock_mhz{0.0f};
        bool is_throttling{false};
    };

    void update(const MeasuredPerformance& perf) { last_ = perf; }

    const MeasuredPerformance& last_measurement() const { return last_; }

    OptimizationPlanner::Plan replan(
        const HardwareProfile& /*profile*/,
        const MeasuredPerformance& /*perf*/, ModelKind /*kind*/) const {
        OptimizationPlanner::Plan plan;
        plan.reason = "runtime-profiler-stub";
        return plan;
    }

private:
    MeasuredPerformance last_;
};

}  // namespace ie
