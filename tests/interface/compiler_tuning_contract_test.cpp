#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

struct RecordingPass : Pass {
    std::vector<std::string>* log;
    std::string id;
    Status result;

    RecordingPass(std::vector<std::string>* l, std::string n, Status r = {})
        : log(l), id(std::move(n)), result(std::move(r)) {}

    std::string name() const override { return id; }
    Status run(Graph&, PassContext&) override {
        log->push_back(id);
        return result;
    }
};

TEST(PassManagerTest, StopsAtFirstFailure) {
    std::vector<std::string> log;
    PassManager manager;
    manager.add(std::make_unique<RecordingPass>(&log, "first"));
    manager.add(std::make_unique<RecordingPass>(
        &log, "second", Error::runtime("boom")));
    manager.add(std::make_unique<RecordingPass>(&log, "third"));

    Graph graph;
    PassContext context;
    auto status = manager.run(graph, context);
    ASSERT_FALSE(status.has_value());
    EXPECT_EQ(status.error().code, Error::Code::Runtime);
    EXPECT_EQ(log, (std::vector<std::string>{"first", "second"}));
    EXPECT_EQ(manager.size(), 3u);
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

TEST(PassRegistryTest, RejectsEmptyCreatorAndName) {
    auto& registry = PassRegistry::instance();
    EXPECT_EQ(registry.register_pass("empty-creator", PassRegistry::Creator{}).error().code,
              Error::Code::InvalidArgument);
    EXPECT_EQ(registry.register_pass("", [] { return std::make_unique<StubPass>(nullptr); }).error().code,
              Error::Code::InvalidArgument);
}

TEST(PassRegistryTest, CreateMissReturnsNotFound) {
    auto& registry = PassRegistry::instance();
    auto missing = registry.create("does-not-exist");
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, Error::Code::NotFound);
}

TEST(PassRegistryTest, NullFactoryReturnsRuntime) {
    auto& registry = PassRegistry::instance();
    ASSERT_TRUE(registry.register_pass("null-factory", [] {
        return std::unique_ptr<Pass>{};
    }).has_value());
    auto created = registry.create("null-factory");
    ASSERT_FALSE(created.has_value());
    EXPECT_EQ(created.error().code, Error::Code::Runtime);
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

namespace {

KernelVariant MakeVariant(std::string name) {
    KernelVariant variant;
    variant.name = std::move(name);
    return variant;
}

HardwareProfile CapableProfile() {
    HardwareProfile profile;
    profile.matrix.tensor_to_vector_ratio = 10.0f;
    profile.mem.vram_total = 8192;
    profile.mem.bandwidth_gbps = 100.0f;
    profile.cu.l3_cache_size = 4096;
    profile.caps.set("avx512", true);
    return profile;
}

struct SelectCase {
    const char* name;
    KernelVariant variant;
    bool expect_match;
};

}  // namespace

TEST(KernelVariantRegistryTest, FiltersEveryRequirement) {
    const HardwareProfile profile = CapableProfile();

    KernelVariant forbidden = MakeVariant("forbidden");
    forbidden.req.forbidden_caps = {"avx512"};

    KernelVariant kind_mismatch = MakeVariant("kind-mismatch");
    kind_mismatch.target_kind = ModelKind::TRANSFORMER;

    KernelVariant ratio_below = MakeVariant("ratio-below");
    ratio_below.req.min_tensor_to_vector_ratio = 20.0f;

    KernelVariant ratio_above = MakeVariant("ratio-above");
    ratio_above.req.max_tensor_to_vector_ratio = 5.0f;

    KernelVariant vram = MakeVariant("vram");
    vram.req.min_vram = 16384;

    KernelVariant bandwidth = MakeVariant("bandwidth");
    bandwidth.req.min_bandwidth = 200.0f;

    KernelVariant l3 = MakeVariant("l3");
    l3.req.min_l3_cache = 8192;

    const std::vector<SelectCase> cases{
        {"forbidden_caps", forbidden, false},
        {"target_kind", kind_mismatch, false},
        {"ratio_below_min", ratio_below, false},
        {"ratio_above_max", ratio_above, false},
        {"min_vram", vram, false},
        {"min_bandwidth", bandwidth, false},
        {"min_l3_cache", l3, false},
        {"all_satisfied", MakeVariant("match"), true},
    };

    for (const SelectCase& test_case : cases) {
        KernelVariantRegistry registry;
        registry.register_variant(OpType::MatMul, DType::F32, test_case.variant);
        auto selected = registry.select(OpType::MatMul, DType::F32,
                                        Shape{{2, 2}}, profile,
                                        ModelKind::TRADITIONAL);
        ASSERT_TRUE(selected.has_value()) << test_case.name;
        EXPECT_EQ(selected->size(), test_case.expect_match ? 1u : 0u)
            << test_case.name;
    }
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
