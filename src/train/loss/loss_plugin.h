#pragma once

#include <string>

#include "core/memory/tensor.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Loss function plugin (docs/readme.md §9.1).
class LossPlugin {
public:
    virtual ~LossPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::string loss_name() const = 0;

    virtual Tensor compute(const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
};

}  // namespace ie
