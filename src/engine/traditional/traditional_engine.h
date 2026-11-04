#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/config/model_format.h"
#include "core/memory/tensor.h"

namespace ie {

// Traditional (non-Transformer) engine contract (docs/readme.md §8). P0.5
// freezes the surface only; the traditional runtime lands later.
class TraditionalEngine {
public:
    Status load_model(const std::string& path, ModelFormat format) {
        (void)path;
        (void)format;
        return Error::unsupported(
            "traditional engine not implemented in P0.5");
    }

    Result<Tensor> infer(const std::vector<Tensor>& inputs) {
        (void)inputs;
        return Error::unsupported(
            "traditional engine not implemented in P0.5");
    }

    Result<std::vector<Tensor>> infer_batch(
        const std::vector<std::vector<Tensor>>& batches) {
        (void)batches;
        return Error::unsupported(
            "traditional engine not implemented in P0.5");
    }
};

}  // namespace ie
