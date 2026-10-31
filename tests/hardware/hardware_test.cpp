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
