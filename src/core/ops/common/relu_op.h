#pragma once

#include <cstdint>

#include "core/ops/op.h"

namespace ie {

// Elementwise ReLU. Consumes one F32 tensor and produces a tensor with the
// same shape.
class ReluOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (inputs.size() != 1 || outputs.size() != 1 ||
            outputs[0] == nullptr) {
            return Error::invalid_argument(
                "Relu expects one input and one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("Relu requires an allocator");
        }
        if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
            return valid.error();
        }
        auto result = Tensor::allocate(inputs[0]->shape, inputs[0]->dtype,
                                       context.allocator);
        if (!result.has_value()) {
            return result.error();
        }
        const float* in = inputs[0]->data<float>();
        float* out = result->data<float>();
        const std::int64_t n = inputs[0]->numel();
        if (context.kernels != nullptr && context.kernels->relu != nullptr) {
            context.kernels->relu(in, out, n);
        } else {
            for (std::int64_t i = 0; i < n; ++i) {
                out[i] = in[i] > 0.0f ? in[i] : 0.0f;
            }
        }
        *outputs[0] = std::move(*result);
        return {};
    }
};

}  // namespace ie
