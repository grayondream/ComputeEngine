#pragma once

#include "core/ops/op.h"

namespace ie {

class MatMulOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (inputs.size() != 2 || outputs.empty()) {
            return Error::invalid_argument(
                "MatMul expects two inputs and one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("MatMul requires an allocator");
        }
        if (inputs[0]->shape.dims.size() != 2 ||
            inputs[1]->shape.dims.size() != 2) {
            return Error::invalid_argument("MatMul supports 2D inputs only");
        }
        const int64_t m = inputs[0]->shape.dims[0];
        const int64_t k = inputs[0]->shape.dims[1];
        const int64_t k2 = inputs[1]->shape.dims[0];
        const int64_t n = inputs[1]->shape.dims[1];
        if (k != k2) {
            return Error::invalid_argument("MatMul inner dimensions mismatch");
        }
        auto result = Tensor::allocate(Shape{{m, n}}, inputs[0]->dtype,
                                       *context.allocator);
        if (!result.has_value()) {
            return result.error();
        }
        const float* a = inputs[0]->data<float>();
        const float* b = inputs[1]->data<float>();
        float* out = result->data<float>();
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < k; ++t) {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        *outputs[0] = std::move(*result);
        return {};
    }
};

}  // namespace ie
