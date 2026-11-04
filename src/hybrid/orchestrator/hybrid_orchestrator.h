#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/tensor.h"

namespace ie {

struct PipelineSpec {
    std::vector<std::string> stages;
};

struct Input {
    std::unordered_map<std::string, Tensor> tensors;
};

struct Output {
    std::vector<Tensor> tensors;
};

// Hybrid orchestration surface (docs/readme.md §7). Pipeline construction and
// execution are not implemented in P0.5.
class HybridOrchestrator {
public:
    Result<Graph> build_pipeline(const PipelineSpec&) {
        return Error::unsupported("hybrid pipeline not implemented in P0.5");
    }

    Result<Output> run(const Graph&, const Input&) {
        return Error::unsupported("hybrid orchestration not implemented in P0.5");
    }
};

}  // namespace ie
