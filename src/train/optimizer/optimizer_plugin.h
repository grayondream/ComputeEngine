#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/memory/allocator.h"
#include "core/memory/tensor.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Optimizer hyper-parameters shared by all optimizer plugins (docs/readme.md
// §9.1). Concrete optimizers arrive with the training milestone.
struct OptimizerConfig {
    float lr{1e-3f};
    float weight_decay{0.0f};
    float beta1{0.9f};
    float beta2{0.999f};
    float eps{1e-8f};
    float momentum{0.0f};
};

class OptimizerPlugin {
public:
    virtual ~OptimizerPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::string optimizer_name() const = 0;

    virtual Status step(std::vector<Tensor*>& params,
                        std::vector<Tensor*>& grads,
                        const OptimizerConfig& config) = 0;
    virtual Status zero_grad(std::vector<Tensor*>& grads) = 0;

    virtual Status save_state(BufferHandle buffer) = 0;
    virtual Status load_state(BufferHandle buffer) = 0;
};

}  // namespace ie
