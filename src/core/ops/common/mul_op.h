#pragma once

#include <variant>

#include "core/ops/op.h"

namespace ie {

class MulOp final : public Op {
public:
    Status configure(const Attrs& attrs) override {
        auto it = attrs.find("scalar");
        if (it == attrs.end()) {
            return Error::invalid_argument("Mul requires a scalar attr");
        }
        if (const auto* value = std::get_if<double>(&it->second)) {
            scalar_ = *value;
        } else if (const auto* ivalue = std::get_if<int64_t>(&it->second)) {
            scalar_ = static_cast<double>(*ivalue);
        } else {
            return Error::invalid_argument("Mul scalar attr must be numeric");
        }
        has_scalar_ = true;
        return {};
    }

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (outputs.empty()) {
            return Error::invalid_argument("Mul expects one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("Mul requires an allocator");
        }
        if (inputs.size() == 2) {
            if (inputs[0]->shape != inputs[1]->shape) {
                return Error::invalid_argument("Mul shape mismatch");
            }
            auto result = Tensor::allocate(inputs[0]->shape, inputs[0]->dtype,
                                           *context.allocator);
            if (!result.has_value()) {
                return result.error();
            }
            const float* a = inputs[0]->data<float>();
            const float* b = inputs[1]->data<float>();
            float* out = result->data<float>();
            const int64_t count = inputs[0]->numel();
            for (int64_t i = 0; i < count; ++i) {
                out[i] = a[i] * b[i];
            }
            *outputs[0] = std::move(*result);
            return {};
        }
        if (inputs.size() == 1 && has_scalar_) {
            auto result = Tensor::allocate(inputs[0]->shape, inputs[0]->dtype,
                                           *context.allocator);
            if (!result.has_value()) {
                return result.error();
            }
            const float* a = inputs[0]->data<float>();
            float* out = result->data<float>();
            const int64_t count = inputs[0]->numel();
            const float scalar = static_cast<float>(scalar_);
            for (int64_t i = 0; i < count; ++i) {
                out[i] = a[i] * scalar;
            }
            *outputs[0] = std::move(*result);
            return {};
        }
        return Error::invalid_argument("Mul expects two inputs or one scalar");
    }

private:
    double scalar_{0.0};
    bool has_scalar_{false};
};

}  // namespace ie
