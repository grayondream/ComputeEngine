#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct Batch {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
};

// Data loader plugin (docs/readme.md §9.1).
class DataloaderPlugin {
public:
    virtual ~DataloaderPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Status load(const std::string& path) = 0;
    virtual Result<Batch> next() = 0;
    virtual void shuffle(std::uint64_t seed) = 0;
};

}  // namespace ie
