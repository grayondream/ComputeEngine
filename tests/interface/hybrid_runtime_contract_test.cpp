#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "core/backend/backend_plugin.h"
#include "core/backend/engine_config.h"
#include "core/memory/transfer_manager.h"
#include "core/plugin/plugin_registry.h"
#include "core/runtime/iterative_executor.h"
#include "core/scheduler/scheduler_plugin.h"
#include "core/hardware/hardware_probe_plugin.h"
#include "hybrid/hybrid_config.h"
#include "hybrid/memory/hybrid_memory.h"
#include "hybrid/orchestrator/hybrid_orchestrator.h"
#include "hybrid/planner/hybrid_planner.h"
#include "hybrid/scheduler/hybrid_scheduler.h"

using namespace ie;

struct StubIterativeExecutor : IterativeExecutor {
    Status run(const Graph&, const BatchPlan&, ExecutionContext&) override {
        return Error::unsupported("stub");
    }
};

struct StubBackendPlugin : BackendPlugin {
    PluginMeta meta() const override { return {"backend","1.0","0.1",{},""}; }
    Result<std::unique_ptr<Backend>> create(const BackendConfig&) override {
        return Error::unsupported("stub");
    }
};
struct StubSchedulerPlugin : SchedulerPlugin {
    PluginMeta meta() const override { return {"scheduler","1.0","0.1",{},""}; }
    Result<BatchPlan> plan(Graph&, SchedulerContext&) override {
        return Error::unsupported("stub");
    }
};
struct StubHardwareProbePlugin : HardwareProbePlugin {
    PluginMeta meta() const override { return {"probe","1.0","0.1",{},""}; }
    Result<HardwareProfile> probe() override {
        return Error::unsupported("stub");
    }
};

TEST(PluginSurfacesTest, BackendSchedulerProbeRegistries) {
    auto b = PluginRegistry<BackendPlugin>::instance().register_plugin(
        std::make_unique<StubBackendPlugin>());
    auto s = PluginRegistry<SchedulerPlugin>::instance().register_plugin(
        std::make_unique<StubSchedulerPlugin>());
    auto p = PluginRegistry<HardwareProbePlugin>::instance().register_plugin(
        std::make_unique<StubHardwareProbePlugin>());
    EXPECT_TRUE(b.has_value() || b.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(s.has_value() || s.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(p.has_value() || p.error().code == Error::Code::AlreadyExists);
}

TEST(HybridConfigTest, DefaultsAreDisabled) {
    HybridConfig cfg;
    EXPECT_FALSE(cfg.enable_hybrid);
    EXPECT_EQ(cfg.gpu_layers, 0);
    EXPECT_EQ(cfg.strategy, HybridConfig::Strategy::AUTO);
    EXPECT_TRUE(cfg.devices.empty());
    EngineConfig engine_cfg;
    EXPECT_FALSE(engine_cfg.hybrid.enable_hybrid);
}

TEST(HybridContractsTest, DefaultsAreUnsupportedOrEmpty) {
    HybridPlanner planner;
    Graph graph;
    auto planned = planner.plan(graph, HybridConfig{});
    ASSERT_FALSE(planned.has_value());
    EXPECT_EQ(planned.error().code, Error::Code::Unsupported);

    HybridScheduler scheduler;
    scheduler.submit(Request{});
    EXPECT_TRUE(scheduler.plan_next_iteration().prefills.empty());
    ExecutionContext context;
    EXPECT_EQ(scheduler.execute(graph, ExecutionPlan{}, context).error().code,
              Error::Code::Unsupported);

    HybridMemoryManager memory;
    EXPECT_TRUE(memory.configure(HybridConfig{}).has_value());
    EXPECT_EQ(memory.allocate(16, Device::CPU).error().code,
              Error::Code::Unsupported);
    EXPECT_TRUE(memory.release(nullptr).has_value());

    HybridOrchestrator orchestrator;
    EXPECT_EQ(orchestrator.build_pipeline(PipelineSpec{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(orchestrator.run(graph, Input{}).error().code,
              Error::Code::Unsupported);
}

TEST(TransferManagerTest, RoutesAndFallbacks) {
    TransferManager manager;
    bool routed = false;
    BufferHandle seen_dst = nullptr;
    Device seen_dst_dev = Device::VULKAN;
    BufferHandle seen_src = nullptr;
    Device seen_src_dev = Device::VULKAN;
    std::size_t seen_bytes = 0;
    int src_buf = 0;
    int dst_buf = 0;
    manager.register_route(Device::CPU, Device::CUDA,
                           [&routed, &seen_dst, &seen_dst_dev, &seen_src,
                            &seen_src_dev, &seen_bytes](
                               BufferHandle dst, Device dst_dev,
                               BufferHandle src, Device src_dev,
                               std::size_t bytes) {
                               routed = true;
                               seen_dst = dst;
                               seen_dst_dev = dst_dev;
                               seen_src = src;
                               seen_src_dev = src_dev;
                               seen_bytes = bytes;
                               return Status{};
                           });
    EXPECT_TRUE(manager
                    .transfer(&dst_buf, Device::CUDA, &src_buf, Device::CPU,
                              sizeof(int))
                    .has_value());
    EXPECT_TRUE(routed);
    EXPECT_EQ(seen_dst, static_cast<BufferHandle>(&dst_buf));
    EXPECT_EQ(seen_dst_dev, Device::CUDA);
    EXPECT_EQ(seen_src, static_cast<BufferHandle>(&src_buf));
    EXPECT_EQ(seen_src_dev, Device::CPU);
    EXPECT_EQ(seen_bytes, sizeof(int));
    EXPECT_TRUE(manager.transfer(nullptr, Device::CPU, nullptr, Device::CPU, 0)
                    .has_value());
    EXPECT_EQ(manager.transfer(nullptr, Device::CUDA, nullptr, Device::VULKAN, 0)
                  .error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(manager.wait(1).error().code, Error::Code::Unsupported);
}

TEST(TransferManagerTest, SameDeviceMemcpyAndNullChecks) {
    TransferManager manager;
    char src[] = {'a', 'b', 'c', 'd'};
    char dst[sizeof(src)] = {};
    ASSERT_TRUE(manager.transfer(dst, Device::CPU, src, Device::CPU, sizeof(src))
                    .has_value());
    EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);

    EXPECT_EQ(manager.transfer(nullptr, Device::CPU, src, Device::CPU, sizeof(src))
                  .error().code,
              Error::Code::InvalidArgument);
    EXPECT_EQ(manager.transfer(dst, Device::CPU, nullptr, Device::CPU, sizeof(src))
                  .error().code,
              Error::Code::InvalidArgument);
}

TEST(TransferManagerTest, CopyStaysSameDeviceOnlyEvenWithRoute) {
    TransferManager manager;
    int route_calls = 0;
    manager.register_route(Device::CPU, Device::CUDA,
                           [&route_calls](BufferHandle, Device, BufferHandle,
                                          Device, std::size_t) {
                               ++route_calls;
                               return Status{};
                           });
    int src_buf = 0;
    int dst_buf = 0;
    EXPECT_EQ(manager.copy(Device::CPU, Device::CUDA, &src_buf, sizeof(int))
                  .error()
                  .code,
              Error::Code::Unsupported);
    EXPECT_EQ(route_calls, 0);
    EXPECT_TRUE(manager
                    .transfer(&dst_buf, Device::CUDA, &src_buf, Device::CPU,
                              sizeof(int))
                    .has_value());
    EXPECT_EQ(route_calls, 1);

    int value = 42;
    EXPECT_TRUE(manager.copy(Device::CPU, Device::CPU, &value, sizeof(value))
                    .has_value());
    EXPECT_EQ(value, 42);
    EXPECT_EQ(manager.copy(Device::CPU, Device::CPU, nullptr, sizeof(value))
                  .error()
                  .code,
              Error::Code::InvalidArgument);
}

TEST(TransferManagerTest, PrefetchAsyncIsUnsupported) {
    TransferManager manager;
    Stream stream;
    EXPECT_EQ(manager.prefetch_async(nullptr, nullptr, stream, 0).error().code,
              Error::Code::Unsupported);
}

TEST(IterativeExecutorTest, StubReturnsUnsupported) {
    StubIterativeExecutor executor;
    Graph graph;
    BatchPlan plan;
    ExecutionContext context;
    EXPECT_EQ(executor.run(graph, plan, context).error().code,
              Error::Code::Unsupported);
}
