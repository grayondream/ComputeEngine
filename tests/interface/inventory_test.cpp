#include <gtest/gtest.h>

#include <array>
#include <iostream>
#include <iterator>
#include <string_view>

// Task 1: configuration, feature gates, API/plugin version.
#include "core/config/feature_flags.h"
#include "core/config/model_format.h"
#include "core/plugin/plugin_api.h"
#include "core/plugin/plugin_registry.h"
#include "core/backend/engine_config.h"

// Task 2: compiler and tuning.
#include "core/ir/pass.h"
#include "core/ir/pass_manager.h"
#include "core/ir/pass_plugin.h"
#include "core/ir/pass_registry.h"
#include "core/tuning/hardware_aware_cache.h"
#include "core/tuning/kernel.h"
#include "core/tuning/kernel_variant.h"
#include "core/tuning/kernel_variant_registry.h"
#include "core/tuning/runtime_autotuner.h"
#include "core/tuning/runtime_hardware_profiler.h"

// Task 3: runtime, transfer, and hybrid.
#include "core/backend/backend_plugin.h"
#include "core/hardware/hardware_probe_plugin.h"
#include "core/memory/transfer_manager.h"
#include "core/runtime/iterative_executor.h"
#include "core/scheduler/scheduler_plugin.h"
#include "hybrid/hybrid_config.h"
#include "hybrid/memory/hybrid_memory.h"
#include "hybrid/orchestrator/hybrid_orchestrator.h"
#include "hybrid/planner/hybrid_planner.h"
#include "hybrid/scheduler/hybrid_scheduler.h"

// Task 4: rule engine.
#include "ops/rule/rule_ops.h"
#include "rules/agenda/agenda.h"
#include "rules/fact/fact.h"
#include "rules/rule/rule.h"
#include "rules/rule_engine.h"

// Task 5: op families and experimental ops.
#include "core/ir/op_plugin.h"
#include "core/ops/unimplemented_op.h"
#include "experimental/attention/flash_attention.h"
#include "experimental/moe/moe_dispatch.h"
#include "ops/control/control_ops.h"
#include "ops/traditional/traditional_ops.h"
#include "ops/transformer/transformer_ops.h"

// Task 6: LLM and engine interfaces.
#include "engine/traditional/traditional_engine.h"
#include "experimental/kv_cache/paged_kv_cache.h"
#include "experimental/speculative/speculative_decoder.h"
#include "llm/model_loader/model_loader.h"
#include "llm/sampler/sampler.h"
#include "llm/streaming/streaming.h"
#include "llm/tokenizer/tokenizer.h"
#include "llm/transformer/transformer_model.h"

// Task 7: training.
#include "train/amp/amp.h"
#include "train/autograd/autograd_engine.h"
#include "train/checkpoint/checkpoint.h"
#include "train/data/dataloader_plugin.h"
#include "train/distributed/distributed.h"
#include "train/loss/loss_plugin.h"
#include "train/optimizer/optimizer_plugin.h"
#include "train/scheduler/lr_scheduler_plugin.h"
#include "train/trainer/trainer.h"

// Task 8: validation.
#include "validation/comparator/comparator.h"
#include "validation/config.h"
#include "validation/cross_framework/cross_framework.h"
#include "validation/determinism/determinism.h"
#include "validation/reference_tracer/reference_tracer.h"
#include "validation/reference_tracer/trace_ir.h"
#include "validation/report/report.h"
#include "validation/tensor_hash/tensor_hash.h"

// Task 9: converter and quantization.
#include "converter/format_loader/format_loader.h"
#include "converter/model_converter.h"
#include "converter/op_mapping/op_mapping.h"
#include "converter/quantization/quant_plugins.h"
#include "converter/validation/conversion_validator.h"
#include "core/plugin/quant_plugin.h"

// Task 10: C ABI and support layers.
#include "api/c_api/ie_api.h"
#include "api/cli/cli.h"
#include "api/grpc/grpc_service.h"
#include "api/openai_compat/openai_compat.h"
#include "benchmarks/benchmark_runner.h"
#include "fault_tolerance/fault_tolerance.h"
#include "observability/observability.h"
#include "packaging/packaging.h"
#include "sdk/sdk.h"
#include "security/security.h"
#include "tools/tools.h"

namespace {

struct InterfaceGroup {
    std::string_view name;
    std::size_t header_count;
};

// One group per P0.5 task; the inventory document has one row per header.
constexpr std::array<InterfaceGroup, 10> kInterfaceGroups{{
    {"Task 1: configuration, feature gates, API/plugin version", 5},
    {"Task 2: compiler and tuning", 10},
    {"Task 3: runtime, transfer, and hybrid", 10},
    {"Task 4: rule engine", 5},
    {"Task 5: op families and experimental ops", 7},
    {"Task 6: LLM and engine interfaces", 8},
    {"Task 7: training", 9},
    {"Task 8: validation", 8},
    {"Task 9: converter and quantization", 6},
    {"Task 10: C ABI and support layers", 11},
}};

}  // namespace

TEST(InterfaceInventoryTest, TenContractGroupsAreComplete) {
    EXPECT_EQ(std::size(kInterfaceGroups), 10u);
    for (const auto& group : kInterfaceGroups) {
        EXPECT_FALSE(group.name.empty());
        EXPECT_GT(group.header_count, 0u);
        std::cout << group.name << " (" << group.header_count << " headers)\n";
    }
}
