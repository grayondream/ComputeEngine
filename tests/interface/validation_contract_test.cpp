#include <gtest/gtest.h>

#include "core/backend/engine_config.h"
#include "validation/comparator/comparator.h"
#include "validation/config.h"
#include "validation/cross_framework/cross_framework.h"
#include "validation/determinism/determinism.h"
#include "validation/reference_tracer/reference_tracer.h"
#include "validation/report/report.h"
#include "validation/tensor_hash/tensor_hash.h"

using namespace ie;

TEST(ValidationConfigTest, DefaultsAreOff) {
    ValidationConfig config;
    EXPECT_FALSE(config.enabled);
    EXPECT_FALSE(config.tracer.enable_pytorch_trace);
    EXPECT_FALSE(config.hasher.enable);
    EXPECT_EQ(config.hasher.hash_type, TensorHasher::HashType::XXH64);
    EXPECT_FALSE(config.determinism.enabled);
    EngineConfig engine;
    EXPECT_FALSE(engine.validation.enabled);
}

TEST(ValidationContractsTest, DefaultsAreUnsupported) {
    ReferenceTracer tracer;
    EXPECT_EQ(tracer.trace_onnx("m", {}).error().code, Error::Code::Unsupported);
    Tensor tensor;
    EXPECT_EQ(TensorHasher::hash_tensor(tensor).error().code,
              Error::Code::Unsupported);
    DeterminismGuard guard;
    EXPECT_EQ(guard.enable(DeterminismConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(Comparator::compare(TraceIR{}, TraceIR{}, Comparator::CompareConfig{})
                  .error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(CrossFrameworkAdapter::from_vllm("dir").error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(ReportGenerator::to_json(ValidationReport{}, "x").error().code,
              Error::Code::Unsupported);
}
