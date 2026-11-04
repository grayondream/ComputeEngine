#pragma once

#include <string>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct QuantConfig {
    std::string format;
    int bits{8};
    bool symmetric{true};
};

// Quantization plugin contract (docs/readme.md §11.5). Plugins are frozen at
// api_version "0.1"; implementations land after P0.5.
class QuantPlugin {
public:
    virtual ~QuantPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Result<Tensor> quantize(const Tensor& tensor,
                                    const QuantConfig& config) = 0;
    virtual Result<Tensor> dequantize(const Tensor& tensor,
                                      const QuantConfig& config) = 0;
};

}  // namespace ie
