#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "validation/reference_tracer/trace_ir.h"

namespace ie {

// Captures reference traces from external frameworks. Not implemented in
// P0.5: every entry point reports Unsupported.
class ReferenceTracer {
public:
    Result<TraceIR> trace_pytorch(const std::string& model_path,
                                  const std::vector<Tensor>& inputs) {
        (void)model_path;
        (void)inputs;
        return Error::unsupported("reference tracer not implemented in P0.5");
    }

    Result<TraceIR> trace_onnx(const std::string& model_path,
                               const std::vector<Tensor>& inputs) {
        (void)model_path;
        (void)inputs;
        return Error::unsupported("reference tracer not implemented in P0.5");
    }

    Result<TraceIR> load(const std::string& path) {
        (void)path;
        return Error::unsupported("reference tracer not implemented in P0.5");
    }

    Status save(const TraceIR& trace, const std::string& path) {
        (void)trace;
        (void)path;
        return Error::unsupported("reference tracer not implemented in P0.5");
    }
};

}  // namespace ie
