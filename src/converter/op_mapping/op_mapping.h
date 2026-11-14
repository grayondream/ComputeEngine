#pragma once

#include <string_view>

#include "core/base/error.h"
#include "core/ir/op_type.h"

namespace ie {

// Maps a source-framework operator name onto the frozen OpType enum
// (docs/readme.md §11.4).
class OpMapping {
public:
    virtual ~OpMapping() = default;

    virtual Result<OpType> map(std::string_view name) = 0;
};

class OnnxMapping final : public OpMapping {
public:
    Result<OpType> map(std::string_view name) override;
};

class TfliteMapping final : public OpMapping {
public:
    Result<OpType> map(std::string_view /*name*/) override {
        return Error::unsupported("tflite op mapping not implemented in P0.5");
    }
};

class PyTorchMapping final : public OpMapping {
public:
    Result<OpType> map(std::string_view /*name*/) override {
        return Error::unsupported("pytorch op mapping not implemented in P0.5");
    }
};

class GgufMapping final : public OpMapping {
public:
    Result<OpType> map(std::string_view /*name*/) override {
        return Error::unsupported("gguf op mapping not implemented in P0.5");
    }
};

}  // namespace ie
