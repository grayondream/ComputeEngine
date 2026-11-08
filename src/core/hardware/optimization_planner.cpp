#include "core/hardware/optimization_planner.h"

#include <string>

#include "core/base/shape.h"
#include "core/tuning/cpu_kernel_variants.h"

namespace ie {
namespace {

std::string select_kernel_name(OpType op, const HardwareProfile& profile,
                               const char* fallback) {
    const Result<KernelVariant> variant = select_best_cpu_variant(
        op, DType::F32, Shape{}, profile);
    if (!variant.has_value()) {
        return fallback;
    }
    return variant->name;
}

}  // namespace

OptimizationPlanner::Plan OptimizationPlanner::plan(
    const HardwareProfile& profile, const Graph& /*graph*/,
    ModelKind /*kind*/, const EngineConfig& /*config*/) {
    Plan plan;
    plan.preferred_kernels = {
        select_kernel_name(OpType::MatMul, profile, "matmul_scalar"),
        select_kernel_name(OpType::Relu, profile, "relu_scalar"),
        select_kernel_name(OpType::Softmax, profile, "softmax_scalar"),
    };
    plan.reason = "capability-driven:" + plan.preferred_kernels[0];
    return plan;
}

}  // namespace ie
