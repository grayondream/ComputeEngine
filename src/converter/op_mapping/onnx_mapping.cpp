#include "converter/op_mapping/op_mapping.h"

#include <string>
#include <string_view>
#include <unordered_map>

namespace ie {
namespace {

const std::unordered_map<std::string_view, OpType>& onnx_ops() {
    static const std::unordered_map<std::string_view, OpType> ops = {
        {"Add", OpType::Add},
        {"Mul", OpType::Mul},
        {"MatMul", OpType::MatMul},
        {"Gemm", OpType::Gemm},
        {"Relu", OpType::Relu},
        {"Softmax", OpType::Softmax},
        {"Conv", OpType::Conv2D},
        {"ConvTranspose", OpType::ConvTranspose2D},
        {"MaxPool", OpType::MaxPool2D},
        {"AveragePool", OpType::AvgPool2D},
        {"GlobalAveragePool", OpType::GlobalAvgPool},
        {"BatchNormalization", OpType::BatchNorm},
        {"Flatten", OpType::Flatten},
        {"Reshape", OpType::Reshape},
        {"Transpose", OpType::Transpose},
        {"Concat", OpType::Concat},
        {"Slice", OpType::Slice},
        {"Gather", OpType::Gather},
        {"ReduceMean", OpType::Reduce},
        {"ReduceSum", OpType::Reduce},
        {"Pad", OpType::Pad},
        {"Upsample", OpType::Upsample},
        {"Clip", OpType::Clip},
        {"Sigmoid", OpType::Sigmoid},
        {"Tanh", OpType::Tanh},
    };
    return ops;
}

}  // namespace

Result<OpType> OnnxMapping::map(std::string_view name) {
    const auto& ops = onnx_ops();
    const auto found = ops.find(name);
    if (found == ops.end()) {
        return Error::unsupported("onnx: unsupported op '" +
                                  std::string(name) + "'");
    }
    return found->second;
}

}  // namespace ie
