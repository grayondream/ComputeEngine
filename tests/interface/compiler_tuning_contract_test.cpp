#include <gtest/gtest.h>
#include <memory>

#include "core/hardware/profile.h"
#include "core/ir/graph_builder.h"
#include "core/ir/pass_manager.h"
#include "core/ir/pass_plugin.h"
#include "core/ir/pass_registry.h"
#include "core/plugin/plugin_registry.h"
#include "core/tuning/hardware_aware_cache.h"
#include "core/tuning/kernel_variant_registry.h"
#include "core/tuning/runtime_autotuner.h"
#include "core/tuning/runtime_hardware_profiler.h"

using namespace ie;

struct StubPass : Pass {
    int* calls;
    explicit StubPass(int* c) : calls(c) {}
    std::string name() const override { return "stub"; }
    Status run(Graph&, PassContext&) override { ++*calls; return {}; }
};

TEST(PassManagerTest, RunsPassesInOrder) {
    int calls = 0;
    PassManager manager;
    manager.add(std::make_unique<StubPass>(&calls));
    manager.add(std::make_unique<StubPass>(&calls));
    Graph graph;
    PassContext context;
    ASSERT_TRUE(manager.run(graph, context).has_value());
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(manager.size(), 2u);
}

TEST(PassRegistryTest, RegisterCreateAndDuplicate) {
    auto& registry = PassRegistry::instance();
    ASSERT_TRUE(registry.register_pass("stub", [] { return std::make_unique<StubPass>(nullptr); }).has_value());
    auto pass = registry.create("stub");
    ASSERT_TRUE(pass.has_value());
    EXPECT_EQ(pass.value()->name(), "stub");
    auto duplicate = registry.register_pass("stub", [] { return std::make_unique<StubPass>(nullptr); });
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, Error::Code::AlreadyExists);
}

struct StubPassPlugin : PassPlugin {
    PluginMeta meta() const override { return {"pass","1.0","0.1",{},""}; }
    std::unique_ptr<Pass> create() override { return nullptr; }
};

TEST(PassPluginRegistryTest, RegistersCompatiblePlugin) {
    auto& registry = PluginRegistry<PassPlugin>::instance();
    auto status = registry.register_plugin(std::make_unique<StubPassPlugin>());
    EXPECT_TRUE(status.has_value() ||
                status.error().code == Error::Code::AlreadyExists);
}

TEST(GraphBindContractTest, CustomBinderBindsInputs) {
    GraphBuilder builder;
    Graph graph = std::move(builder).add(2).mul(3.0).build();
    auto bound = graph.bind({{"a", Shape{{2}}}, {"b", Shape{{2}}}});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->shapes.size(), 2u);
}

TEST(KernelVariantRegistryTest, SelectsByCapability) {
    KernelVariant variant;
    variant.name = "avx2";
    variant.target_kind = ModelKind::TRADITIONAL;
    variant.req.required_caps = {"avx2"};
    KernelVariantRegistry registry;
    registry.register_variant(OpType::MatMul, DType::F32, variant);

    HardwareProfile with_avx2;
    with_avx2.caps.set("avx2", true);
    auto selected = registry.select(OpType::MatMul, DType::F32, Shape{{2, 2}},
                                    with_avx2, ModelKind::TRADITIONAL);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->size(), 1u);

    HardwareProfile without;
    auto none = registry.select(OpType::MatMul, DType::F32, Shape{{2, 2}},
                                without, ModelKind::TRADITIONAL);
    ASSERT_TRUE(none.has_value());
    EXPECT_TRUE(none->empty());
}

TEST(TuningContractsTest, DefaultsAreUnsupportedOrDefined) {
    RuntimeAutotuner tuner;
    EXPECT_EQ(tuner.tune(OpType::MatMul, DType::F32, Shape{{1}},
                         HardwareProfile{}, ModelKind::UNKNOWN)
                  .error().code,
              Error::Code::Unsupported);

    HardwareAwareCache cache;
    HardwareAwareCache::Key key;
    EXPECT_EQ(cache.get(key).error().code, Error::Code::Unsupported);

    RuntimeHardwareProfiler profiler;
    profiler.update(RuntimeHardwareProfiler::MeasuredPerformance{});
    auto plan = profiler.replan(HardwareProfile{}, {}, ModelKind::UNKNOWN);
    EXPECT_EQ(plan.reason, "runtime-profiler-stub");
}
