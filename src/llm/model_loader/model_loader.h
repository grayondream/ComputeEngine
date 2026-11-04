#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "core/base/error.h"
#include "core/config/model_format.h"
#include "core/ir/graph.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct ModelConfig {
    std::string path;
    ModelFormat format{ModelFormat::CUSTOM};
    int max_seq_len{0};
};

struct ModelLoaderConfig {
    std::string path;
    ModelFormat format{ModelFormat::CUSTOM};
    std::unordered_map<std::string, std::string> options;
};

// Model loader contract (docs/readme.md §8.1). Loaders turn a serialized model
// into the frozen Graph IR.
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    virtual Result<Graph> load(const ModelConfig& config) = 0;
};

class ModelLoaderPlugin {
public:
    virtual ~ModelLoaderPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::unique_ptr<ModelLoader> create() = 0;
};

}  // namespace ie
