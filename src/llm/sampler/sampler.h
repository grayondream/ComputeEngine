#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "core/base/error.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct SamplingParams {
    float temperature{1.0f};
    int top_k{0};
    float top_p{1.0f};
    std::uint64_t seed{0};
};

// Sampler contract (docs/readme.md §8.3). Samplers pick one token id from the
// logits slice under the given decoding parameters.
class Sampler {
public:
    virtual ~Sampler() = default;

    virtual int sample(std::span<const float> logits,
                       const SamplingParams& params) = 0;
};

class SamplerPlugin {
public:
    virtual ~SamplerPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::unique_ptr<Sampler> create() = 0;
};

}  // namespace ie
