#include <gtest/gtest.h>

#include <string>
#include <vector>

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

using namespace ie;

TEST(CApiTest, VersionAndBackendLifecycle) {
    EXPECT_EQ(std::string(ie_api_version()), "0.1");
    EXPECT_EQ(IE_API_VERSION_MAJOR, 0);
    EXPECT_EQ(IE_API_VERSION_MINOR, 1);

    Backend* backend = ie_create_backend("cpu", nullptr);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->device(), Device::CPU);
    ie_destroy_backend(backend);
    ie_destroy_backend(nullptr);

    EXPECT_EQ(ie_create_backend("unknown", nullptr), nullptr);
    EXPECT_EQ(ie_create_backend(nullptr, nullptr), nullptr);
}

TEST(SupportContractsTest, DefaultsAreDefined) {
    AuditLog audit;
    audit.record("event");
    EXPECT_EQ(audit.entries().size(), 1u);
    audit.record(AuditEntry{"login", "alice", "ok", 42});
    ASSERT_EQ(audit.entries().size(), 2u);
    EXPECT_EQ(audit.entries().back().actor, "alice");

    TraceExporter exporter;
    exporter.begin_event("e", "cat");
    exporter.instant_event("i", "cat");
    exporter.counter("c", 1.0);
    exporter.set_device(Device::CPU);
    exporter.set_stream(0);
    exporter.set_phase("test");
    exporter.set_model_kind(ModelKind::TRADITIONAL);
    exporter.end_event();
    EXPECT_EQ(exporter.flush("x").error().code, Error::Code::Unsupported);

    GraphExporter::to_dot(Graph{}, "x");
    EXPECT_EQ(GraphExporter::to_json(Graph{}, "x").error().code,
              Error::Code::Unsupported);
    MetricsExporter metrics;
    EXPECT_EQ(metrics.to_json("x").error().code, Error::Code::Unsupported);
    EXPECT_EQ(metrics.to_prometheus("x").error().code,
              Error::Code::Unsupported);

    BenchmarkRunner runner;
    EXPECT_EQ(runner.run(BenchmarkCase{"case"}).error().code,
              Error::Code::Unsupported);
}

TEST(ApiStubContractsTest, ServersAndCliReturnUnsupported) {
    OpenAIServer openai;
    EXPECT_EQ(openai.start(OpenAIConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(openai.run().error().code, Error::Code::Unsupported);
    EXPECT_EQ(openai.stop().error().code, Error::Code::Unsupported);

    GrpcService grpc;
    EXPECT_EQ(grpc.start(GrpcConfig{}).error().code, Error::Code::Unsupported);
    EXPECT_EQ(grpc.run().error().code, Error::Code::Unsupported);
    EXPECT_EQ(grpc.stop().error().code, Error::Code::Unsupported);

    CliApp cli;
    EXPECT_EQ(cli.start().error().code, Error::Code::Unsupported);
    EXPECT_EQ(cli.run(0, nullptr).error().code, Error::Code::Unsupported);
    EXPECT_EQ(cli.stop().error().code, Error::Code::Unsupported);
}

TEST(SecurityContractsTest, TenantsSandboxAndModelSecurity) {
    TenantManager tenants;
    EXPECT_EQ(tenants.create_tenant(TenantConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_FALSE(tenants.exists("tenant"));
    EXPECT_TRUE(tenants.list().empty());

    Sandbox sandbox;
    EXPECT_EQ(sandbox.initialize(SandboxConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(sandbox.enter().error().code, Error::Code::Unsupported);
    EXPECT_EQ(sandbox.exit().error().code, Error::Code::Unsupported);
    EXPECT_EQ(sandbox.execute("true").error().code,
              Error::Code::Unsupported);

    ModelSecurity security;
    EXPECT_EQ(security.verify_license("model").error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(security.inspect_license("model").error().code,
              Error::Code::Unsupported);
}

TEST(FaultToleranceContractsTest, HealthAndResilienceDefaults) {
    HealthMonitor monitor;
    EXPECT_TRUE(monitor.start().has_value());
    EXPECT_EQ(monitor.status().state, HealthState::HEALTHY);
    EXPECT_TRUE(monitor.stop().has_value());

    ResiliencePolicy policy;
    ResilienceConfig config;
    config.max_retries = 5;
    EXPECT_TRUE(policy.configure(config).has_value());
    EXPECT_EQ(policy.config().max_retries, 5);
    EXPECT_EQ(policy.evaluate("backend", 0).error().code,
              Error::Code::Unsupported);

    DebugTools tools;
    EXPECT_TRUE(tools.start_profiling().has_value());
    EXPECT_EQ(tools.dump_graph(Graph{}, "x").error().code,
              Error::Code::Unsupported);
    EXPECT_TRUE(tools.stop_profiling().has_value());
}

TEST(ObservabilityContractsTest, MetricsLoggerTracerProfiler) {
    MetricsRegistry registry;
    registry.increment("requests");
    registry.observe("latency", 1.0);
    EXPECT_TRUE(registry.snapshot().empty());

    Logger logger;
    logger.set_level(LogLevel::DEBUG);
    logger.log(LogLevel::INFO, "message");
    EXPECT_EQ(logger.level(), LogLevel::DEBUG);

    Tracer tracer;
    tracer.begin_span("span", "category");
    tracer.end_span();
    EXPECT_TRUE(tracer.spans().empty());

    Profiler profiler;
    profiler.begin("section");
    profiler.end();
    EXPECT_TRUE(profiler.samples().empty());
}

TEST(ExperimentContractsTest, DataStructuresAndUnsupportedMethods) {
    Experiment experiment;
    experiment.id = "exp";
    experiment.config.model_kind = ModelKind::TRANSFORMER;
    EXPECT_FALSE(experiment.config.features.enable_training);
    EXPECT_EQ(experiment.result.latency_p50, 0.0);

    ExperimentStore store;
    store.save(experiment);
    EXPECT_TRUE(store.list("").empty());
    EXPECT_EQ(store.load("exp").error().code, Error::Code::Unsupported);
    EXPECT_EQ(store.export_csv("x").error().code, Error::Code::Unsupported);

    ExperimentRunner runner;
    EXPECT_EQ(runner.run_matrix(ExperimentRunner::Matrix{}, Graph{}).error().code,
              Error::Code::Unsupported);

    auto diffs = ComparisonReport::compare(experiment, experiment);
    EXPECT_TRUE(diffs.empty());
    EXPECT_EQ(ComparisonReport::to_markdown({}, "x").error().code,
              Error::Code::Unsupported);
}

TEST(SdkAndPackagingContractsTest, MetadataDefaults) {
    SdkInfo info = sdk_info();
    EXPECT_EQ(info.api_version, "0.1");
    EXPECT_FALSE(info.name.empty());

    PackageInfo package;
    package.name = "engine";
    EXPECT_EQ(Packager::produce(package, "out").error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(Packager::inspect("engine.tar.gz").error().code,
              Error::Code::Unsupported);
}
