#pragma once

#include "core/ops/op.h"

namespace ie {

class AddOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (inputs.size() != 2 || outputs.empty()) {
            return Error::invalid_argument("Add expects two inputs and one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("Add requires an allocator");
        }
        if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
            return valid.error();
        }
        if (inputs[0]->shape != inputs[1]->shape) {
            return Error::invalid_argument("Add shape mismatch");
        }
        auto result = Tensor::allocate(inputs[0]->shape, inputs[0]->dtype,
                                       context.allocator);
        if (!result.has_value()) {
            return result.error();
        }
        const float* a = inputs[0]->data<float>();
        const float* b = inputs[1]->data<float>();
        float* out = result->data<float>();
        const int64_t count = inputs[0]->numel();
        for (int64_t i = 0; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
        *outputs[0] = std::move(*result);
        return {};
    }
};

}  // namespace ie
