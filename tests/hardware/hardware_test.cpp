#include <gtest/gtest.h>

#include "core/hardware/detector.h"
#include "core/hardware/optimization_planner.h"

using namespace ie;

TEST(CapabilitySetTest, SetHasList) {
    CapabilitySet c;
    EXPECT_FALSE(c.has("avx2"));
    c.set("avx2", true);
    EXPECT_TRUE(c.has("avx2"));
    c.set("avx2", false);
    EXPECT_FALSE(c.has("avx2"));
}

TEST(HardwareDetectorTest, DetectCpuReturnsProfileWithFingerprint) {
    auto p = HardwareDetector::detect_cpu();
    ASSERT_TRUE(p.has_value());
    EXPECT_FALSE(p->fingerprint.empty());
    EXPECT_FALSE(p->vendor.empty());
    EXPECT_GE(p->cu.num_cores, 0);
}

TEST(HardwareDetectorTest, DetectCudaIsUnsupportedInP0) {
    auto p = HardwareDetector::detect_cuda(0);
    ASSERT_FALSE(p.has_value());
    EXPECT_EQ(p.error().code, Error::Code::Unsupported);
}

TEST(OptimizationPlannerTest, ReturnsConservativePlan) {
    EngineConfig cfg;
    Graph g;
    auto plan = OptimizationPlanner::plan(HardwareProfile{}, g,
                                          ModelKind::UNKNOWN, cfg);
    EXPECT_FALSE(plan.reason.empty());
}

TEST(OptimizationPlannerTest, SelectsKernelsFromCaps) {
    EngineConfig cfg;
    Graph g;
    HardwareProfile scalar;
    auto p0 = OptimizationPlanner::plan(scalar, g, ModelKind::UNKNOWN, cfg);
    ASSERT_EQ(p0.preferred_kernels.size(), 3u);
    EXPECT_EQ(p0.preferred_kernels[0], "matmul_scalar");
    EXPECT_EQ(p0.preferred_kernels[1], "relu_scalar");
    EXPECT_EQ(p0.preferred_kernels[2], "softmax_scalar");
    EXPECT_EQ(p0.reason.rfind("capability-driven:", 0), 0u);

    HardwareProfile avx2;
    avx2.caps.set("avx2", true);
    avx2.caps.set("fma", true);
    auto p2 = OptimizationPlanner::plan(avx2, g, ModelKind::UNKNOWN, cfg);
    ASSERT_EQ(p2.preferred_kernels.size(), 3u);
    EXPECT_EQ(p2.preferred_kernels[0], "matmul_avx2");
    EXPECT_EQ(p2.preferred_kernels[1], "relu_avx2");
    EXPECT_EQ(p2.preferred_kernels[2], "softmax_avx2");

    HardwareProfile avx512;
    avx512.caps.set("avx2", true);
    avx512.caps.set("fma", true);
    avx512.caps.set("avx512f", true);
    avx512.caps.set("avx512bw", true);
    avx512.caps.set("avx512vl", true);
    auto p5 = OptimizationPlanner::plan(avx512, g, ModelKind::UNKNOWN, cfg);
    ASSERT_EQ(p5.preferred_kernels.size(), 3u);
    EXPECT_EQ(p5.preferred_kernels[0], "matmul_avx512");
    EXPECT_EQ(p5.preferred_kernels[1], "relu_avx512");
    EXPECT_EQ(p5.preferred_kernels[2], "softmax_avx512");
}

TEST(CpuProfileBuildTest, MirrorsFeaturesAndTopology) {
    CpuFeatures f;
    f.sse2 = f.avx = f.avx2 = f.fma = true;
    f.avx512f = f.avx512bw = f.avx512vl = f.avx512_vnni = true;
    f.amx_bf16 = true;
    CpuTopology t;
    t.vendor = "amd";
    t.device_name = "test-cpu";
    t.num_cores = 6;
    t.threads_per_core = 2;
    t.max_threads_per_core = 2;
    t.l2_cache_size = 3u << 20;
    t.l3_cache_size = 32u << 20;
    t.num_memory_channels = 2;
    t.total_memory = 16ull << 30;
    t.available_memory = 8ull << 30;
    t.numa_nodes = 1;

    const HardwareProfile p = HardwareDetector::build_cpu_profile(f, t);
    EXPECT_EQ(p.vendor, "amd");
    EXPECT_EQ(p.device_name, "test-cpu");
    EXPECT_EQ(p.cu.num_cores, 6);
    EXPECT_EQ(p.cu.threads_per_core, 2);
    EXPECT_EQ(p.cu.l2_cache_size, 3u << 20);
    EXPECT_EQ(p.cu.l3_cache_size, 32u << 20);
    EXPECT_EQ(p.mem.vram_total, 16ull << 30);
    EXPECT_EQ(p.mem.vram_available, 8ull << 30);
    EXPECT_EQ(p.deploy.numa_nodes, 1);
    EXPECT_TRUE(p.matrix.has_sse2);
    EXPECT_TRUE(p.matrix.has_avx2);
    EXPECT_TRUE(p.matrix.has_avx512_vnni);
    EXPECT_TRUE(p.matrix.has_amx_bf16);
    EXPECT_TRUE(p.caps.has("avx2"));
    EXPECT_TRUE(p.caps.has("avx512vnni"));
    EXPECT_TRUE(p.caps.has("amx-bf16"));
    EXPECT_TRUE(p.caps.has("fma"));
    EXPECT_TRUE(p.matrix.has_vector_unit);
    EXPECT_TRUE(p.matrix.has_matrix_unit);
    EXPECT_EQ(p.matrix.matrix_unit_generation, 2);
    EXPECT_FALSE(p.fingerprint.empty());
}

TEST(CpuProfileBuildTest, DisabledFeaturesHaveNoCapsOrFlags) {
    CpuFeatures f;
    CpuTopology t;
    t.num_cores = 1;
    t.threads_per_core = 1;
    t.numa_nodes = 1;
    const HardwareProfile p = HardwareDetector::build_cpu_profile(f, t);
    EXPECT_FALSE(p.matrix.has_sse2);
    EXPECT_FALSE(p.matrix.has_avx2);
    EXPECT_FALSE(p.caps.has("avx2"));
    EXPECT_FALSE(p.matrix.has_vector_unit);
    EXPECT_FALSE(p.matrix.has_matrix_unit);
    EXPECT_EQ(p.matrix.matrix_unit_generation, 0);
    EXPECT_EQ(p.cu.num_cores, 1);
}

TEST(HardwareDetectorTest, DetectCpuIsConsistentAndStable) {
    auto first = HardwareDetector::detect_cpu();
    auto second = HardwareDetector::detect_cpu();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->fingerprint, second->fingerprint);
    EXPECT_GT(first->cu.num_cores, 0);
    EXPECT_EQ(first->matrix.has_sse2, first->caps.has("sse2"));
    EXPECT_EQ(first->matrix.has_avx2, first->caps.has("avx2"));
    EXPECT_EQ(first->matrix.has_avx512_vnni,
              first->caps.has("avx512vnni"));
}
