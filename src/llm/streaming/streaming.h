#pragma once

#include <cstddef>
#include <functional>
#include <string_view>

#include "core/base/error.h"
#include "llm/model_loader/model_loader.h"
#include "llm/sampler/sampler.h"

namespace ie {

struct GenerationParams {
    SamplingParams sampling;
    int max_new_tokens{0};
};

// LLM engine contract (docs/readme.md §8.3). generate() streams decoded text
// through on_token as tokens are produced.
class LLMEngine {
public:
    virtual ~LLMEngine() = default;

    virtual Status load(const ModelConfig& config) = 0;
    virtual Status generate(std::string_view prompt,
                            const GenerationParams& params,
                            std::function<void(std::string_view)> on_token) = 0;
};

struct StreamingConfig {
    bool enabled{false};
    std::size_t buffer_size{0};
};

class TokenStream {
public:
    virtual ~TokenStream() = default;

    virtual void on_token(std::string_view token) = 0;
    virtual void on_end() = 0;
};

}  // namespace ie
