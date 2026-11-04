#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "core/backend/engine_config.h"
#include "validation/comparator/comparator.h"
#include "validation/config.h"
#include "validation/cross_framework/cross_framework.h"
#include "validation/determinism/determinism.h"
#include "validation/reference_tracer/reference_tracer.h"
#include "validation/report/report.h"
#include "validation/tensor_hash/tensor_hash.h"

using namespace ie;

namespace {

using UnsupportedCheck = std::function<Error::Code()>;

void expect_all_unsupported(
    const std::vector<std::pair<const char*, UnsupportedCheck>>& cases) {
    for (const auto& [name, check] : cases) {
        SCOPED_TRACE(name);
        EXPECT_EQ(check(), Error::Code::Unsupported);
    }
}

}  // namespace

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

TEST(ValidationContractsTest, AllCrossFrameworkAdapterMethodsUnsupported) {
    expect_all_unsupported({
        {"to_trace_ir",
         [] { return CrossFrameworkAdapter::to_trace_ir(EngineOutput{}).error().code; }},
        {"from_vllm",
         [] { return CrossFrameworkAdapter::from_vllm("dir").error().code; }},
        {"from_tensorrt_llm",
         [] { return CrossFrameworkAdapter::from_tensorrt_llm("dir").error().code; }},
        {"from_huggingface",
         [] { return CrossFrameworkAdapter::from_huggingface("dir").error().code; }},
        {"from_onnxruntime",
         [] { return CrossFrameworkAdapter::from_onnxruntime("dir").error().code; }},
        {"full_pipeline",
         [] {
             return CrossFrameworkAdapter::full_pipeline(
                        "engine", "reference", Comparator::CompareConfig{})
                 .error()
                 .code;
         }},
    });
}

TEST(ValidationContractsTest, AllReferenceTracerMethodsUnsupported) {
    expect_all_unsupported({
        {"trace_pytorch",
         [] {
             ReferenceTracer tracer;
             return tracer.trace_pytorch("m", std::vector<Tensor>{}).error().code;
         }},
        {"trace_onnx",
         [] {
             ReferenceTracer tracer;
             return tracer.trace_onnx("m", std::vector<Tensor>{}).error().code;
         }},
        {"load",
         [] {
             ReferenceTracer tracer;
             return tracer.load("trace.json").error().code;
         }},
        {"save",
         [] {
             ReferenceTracer tracer;
             return tracer.save(TraceIR{}, "trace.json").error().code;
         }},
    });
}

TEST(ValidationContractsTest, AllTensorHasherMethodsUnsupported) {
    expect_all_unsupported({
        {"hash_tensor",
         [] { return TensorHasher::hash_tensor(Tensor{}).error().code; }},
        {"hash_model",
         [] {
             return TensorHasher::hash_model(Graph{},
                                             TensorHasher::HashType::XXH64)
                 .error()
                 .code;
         }},
        {"verify_manifest",
         [] {
             return TensorHasher::verify_manifest(Graph{}, HashManifest{})
                 .error()
                 .code;
         }},
    });
}

TEST(ValidationContractsTest, AllDeterminismGuardMethodsUnsupported) {
    expect_all_unsupported({
        {"enable",
         [] {
             DeterminismGuard guard;
             return guard.enable(DeterminismConfig{}).error().code;
         }},
        {"verify_batch_invariance",
         [] {
             DeterminismGuard guard;
             return guard.verify_batch_invariance("prompt").error().code;
         }},
        {"warmup",
         [] {
             DeterminismGuard guard;
             return guard.warmup().error().code;
         }},
    });
}

TEST(ValidationContractsTest, AllComparatorMethodsUnsupported) {
    expect_all_unsupported({
        {"compare",
         [] {
             return Comparator::compare(TraceIR{}, TraceIR{},
                                        Comparator::CompareConfig{})
                 .error()
                 .code;
         }},
        {"compare_tensor",
         [] {
             return Comparator::compare_tensor(Tensor{}, Tensor{},
                                               Comparator::CompareConfig{})
                 .error()
                 .code;
         }},
        {"find_first_divergence",
         [] {
             return Comparator::find_first_divergence(TraceIR{}, TraceIR{})
                 .error()
                 .code;
         }},
    });
}
