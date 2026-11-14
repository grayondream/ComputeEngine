#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/base/error.h"

namespace ie {

// Minimal in-tree representation of the ONNX protobuf subset consumed by the
// traditional-model loader. No third-party protobuf/ONNX dependency is used.
struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type{0};
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<uint8_t> raw_data;
};

struct OnnxAttribute {
    std::string name;
    std::variant<int64_t, double, std::string, OnnxTensor,
                 std::vector<int64_t>, std::vector<double>>
        value{int64_t{0}};
    bool has_value{false};
};

struct OnnxNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttribute> attributes;
};

struct OnnxValueInfo {
    std::string name;
    int32_t elem_type{0};
    std::vector<int64_t> dims;
};

struct OnnxGraph {
    std::vector<OnnxNode> nodes;
    std::vector<OnnxTensor> initializers;
    std::vector<OnnxValueInfo> inputs;
    std::vector<OnnxValueInfo> outputs;
    std::vector<OnnxValueInfo> value_info;
};

struct OnnxModel {
    OnnxGraph graph;
};

// Parses a serialized ONNX ModelProto. Every read is bounds-checked; malformed
// or truncated input returns InvalidArgument, unsupported tensor payloads
// return Unsupported.
Result<OnnxModel> parse_onnx_model(std::span<const uint8_t> bytes);

}  // namespace ie
