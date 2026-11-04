#pragma once

#include <string>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "llm/model_loader/model_loader.h"

namespace ie {

// Loads a serialized model into the frozen Graph IR (docs/readme.md §11.4).
class FormatLoader {
public:
    virtual ~FormatLoader() = default;

    virtual Result<Graph> load(const std::string& path,
                               const ModelLoaderConfig& config) = 0;
};

class OnnxLoader final : public FormatLoader {
public:
    Result<Graph> load(const std::string& /*path*/,
                       const ModelLoaderConfig& /*config*/) override {
        return Error::unsupported("onnx loader not implemented in P0.5");
    }
};

class TfliteLoader final : public FormatLoader {
public:
    Result<Graph> load(const std::string& /*path*/,
                       const ModelLoaderConfig& /*config*/) override {
        return Error::unsupported("tflite loader not implemented in P0.5");
    }
};

class CaffeLoader final : public FormatLoader {
public:
    Result<Graph> load(const std::string& /*path*/,
                       const ModelLoaderConfig& /*config*/) override {
        return Error::unsupported("caffe loader not implemented in P0.5");
    }
};

class SafetensorsLoader final : public FormatLoader {
public:
    Result<Graph> load(const std::string& /*path*/,
                       const ModelLoaderConfig& /*config*/) override {
        return Error::unsupported(
            "safetensors loader not implemented in P0.5");
    }
};

class GgufLoader final : public FormatLoader {
public:
    Result<Graph> load(const std::string& /*path*/,
                       const ModelLoaderConfig& /*config*/) override {
        return Error::unsupported("gguf loader not implemented in P0.5");
    }
};

}  // namespace ie
