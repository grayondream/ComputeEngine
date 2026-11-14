#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/config/model_format.h"
#include "core/memory/tensor.h"

namespace ie {

class Engine;

// Traditional (non-Transformer) engine contract (docs/readme.md §8).
// Loads an ONNX model graph, binds caller inputs by graph order, and runs it
// through the regular Engine/CPU backend with the graph weights as constants.
class TraditionalEngine {
public:
    TraditionalEngine();
    ~TraditionalEngine();

    TraditionalEngine(const TraditionalEngine&) = delete;
    TraditionalEngine& operator=(const TraditionalEngine&) = delete;
    TraditionalEngine(TraditionalEngine&&) noexcept;
    TraditionalEngine& operator=(TraditionalEngine&&) noexcept;

    Status load_model(const std::string& path, ModelFormat format);

    Result<Tensor> infer(const std::vector<Tensor>& inputs);

    Result<std::vector<Tensor>> infer_batch(
        const std::vector<std::vector<Tensor>>& batches);

private:
    std::unique_ptr<Engine> engine_;
    std::vector<std::string> input_names_;
};

}  // namespace ie
