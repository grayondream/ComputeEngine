#include "ops/traditional/activation_ops.h"

#include <cmath>
#include <cstdint>
#include <utility>

namespace ie {

Status SigmoidOp::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Sigmoid expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Sigmoid requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    auto result =
        Tensor::allocate(inputs[0]->shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = inputs[0]->data<float>();
    float* out_data = result->data<float>();
    const int64_t count = inputs[0]->shape.numel();
    for (int64_t i = 0; i < count; ++i) {
        const float value = in_data[i];
        if (value >= 0.0f) {
            out_data[i] = 1.0f / (1.0f + std::exp(-value));
        } else {
            const float exponent = std::exp(value);
            out_data[i] = exponent / (1.0f + exponent);
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status TanhOp::forward(const std::vector<Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Tanh expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Tanh requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    auto result =
        Tensor::allocate(inputs[0]->shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = inputs[0]->data<float>();
    float* out_data = result->data<float>();
    const int64_t count = inputs[0]->shape.numel();
    for (int64_t i = 0; i < count; ++i) {
        out_data[i] = std::tanh(in_data[i]);
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace ie
