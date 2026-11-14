#include "engine/traditional/traditional_engine.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/runtime/engine.h"
#include "models/traditional/onnx/onnx_loader.h"

namespace ie {

TraditionalEngine::TraditionalEngine() = default;
TraditionalEngine::~TraditionalEngine() = default;
TraditionalEngine::TraditionalEngine(TraditionalEngine&&) noexcept = default;
TraditionalEngine& TraditionalEngine::operator=(TraditionalEngine&&) noexcept =
    default;

Status TraditionalEngine::load_model(const std::string& path,
                                     ModelFormat format) {
    if (format != ModelFormat::ONNX) {
        return Error::unsupported(
            "traditional engine only loads ONNX models");
    }

    auto graph = load_onnx_model(path);
    if (!graph.has_value()) {
        return graph.error();
    }

    // The graph stores inputs in declaration order; infer() binds caller
    // tensors to that order.
    std::vector<std::string> input_names;
    for (const Value& value : graph->values) {
        if (value.is_input) {
            input_names.push_back(value.name);
        }
    }

    auto engine = std::make_unique<Engine>(EngineConfig{});
    Status loaded = engine->load(*graph);
    if (!loaded.has_value()) {
        return loaded.error();
    }

    engine_ = std::move(engine);
    input_names_ = std::move(input_names);
    return {};
}

Result<Tensor> TraditionalEngine::infer(const std::vector<Tensor>& inputs) {
    if (engine_ == nullptr) {
        return Error::invalid_argument(
            "traditional engine has no loaded model");
    }
    if (inputs.size() != input_names_.size()) {
        return Error::invalid_argument(
            "traditional engine expected " +
            std::to_string(input_names_.size()) + " inputs, got " +
            std::to_string(inputs.size()));
    }

    std::unordered_map<std::string, Tensor> named_inputs;
    named_inputs.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        named_inputs.emplace(input_names_[i], inputs[i]);
    }

    auto outputs = engine_->infer(named_inputs);
    if (!outputs.has_value()) {
        return outputs.error();
    }
    if (outputs->empty()) {
        return Error::invalid_argument("traditional model produced no outputs");
    }
    return std::move(outputs->front());
}

Result<std::vector<Tensor>> TraditionalEngine::infer_batch(
    const std::vector<std::vector<Tensor>>& batches) {
    std::vector<Tensor> results;
    results.reserve(batches.size());
    for (const std::vector<Tensor>& sample : batches) {
        auto output = infer(sample);
        if (!output.has_value()) {
            return output.error();
        }
        results.push_back(std::move(*output));
    }
    return results;
}

}  // namespace ie
