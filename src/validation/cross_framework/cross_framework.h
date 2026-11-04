#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "validation/comparator/comparator.h"
#include "validation/reference_tracer/trace_ir.h"

namespace ie {

struct EngineOutput {
    std::vector<std::string> names;
    std::vector<Tensor> tensors;
};

// Aligns engine output with external framework traces (docs/readme.md §10.6).
// Not implemented in P0.5: every entry point reports Unsupported.
class CrossFrameworkAdapter {
public:
    static Result<TraceIR> to_trace_ir(const EngineOutput& output) {
        (void)output;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }

    static Result<TraceIR> from_vllm(const std::string& output_dir) {
        (void)output_dir;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }

    static Result<TraceIR> from_tensorrt_llm(const std::string& output_dir) {
        (void)output_dir;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }

    static Result<TraceIR> from_huggingface(const std::string& output_dir) {
        (void)output_dir;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }

    static Result<TraceIR> from_onnxruntime(const std::string& output_dir) {
        (void)output_dir;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }

    static Result<CompareReport> full_pipeline(
        const std::string& engine_output,
        const std::string& reference_output,
        const Comparator::CompareConfig& config) {
        (void)engine_output;
        (void)reference_output;
        (void)config;
        return Error::unsupported(
            "cross framework adapter not implemented in P0.5");
    }
};

}  // namespace ie
