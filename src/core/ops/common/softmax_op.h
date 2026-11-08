#pragma once

#include <cmath>
#include <cstdint>

#include "core/ops/op.h"

namespace ie {
namespace detail {

// Stable row-wise softmax: subtract the row maximum before exponentiating so
// large logits cannot overflow. This is the reference used when no
// capability-selected kernel is injected.
inline void softmax_reference(const float* in, float* out, std::int64_t rows,
                              std::int64_t cols) {
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;
        if (cols <= 0) {
            continue;
        }
        float max_value = row_in[0];
        for (std::int64_t c = 1; c < cols; ++c) {
            if (row_in[c] > max_value) {
                max_value = row_in[c];
            }
        }
        float sum = 0.0f;
        for (std::int64_t c = 0; c < cols; ++c) {
            const float value = std::exp(row_in[c] - max_value);
            row_out[c] = value;
            sum += value;
        }
        const float inv_sum = 1.0f / sum;
        for (std::int64_t c = 0; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

}  // namespace detail

// Row-wise softmax over the last axis. Supports 1D [cols] and 2D
// [rows, cols] inputs; any other rank is rejected.
class SoftmaxOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (inputs.size() != 1 || outputs.size() != 1 ||
            outputs[0] == nullptr) {
            return Error::invalid_argument(
                "Softmax expects one input and one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("Softmax requires an allocator");
        }
        if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
            return valid.error();
        }

        const std::vector<int64_t>& dims = inputs[0]->shape.dims;
        std::int64_t rows = 1;
        std::int64_t cols = 0;
        if (dims.size() == 1) {
            cols = dims[0];
        } else if (dims.size() == 2) {
            rows = dims[0];
            cols = dims[1];
        } else {
            return Error::invalid_argument(
                "Softmax supports 1D and 2D inputs only");
        }

        auto result = Tensor::allocate(inputs[0]->shape, inputs[0]->dtype,
                                       context.allocator);
        if (!result.has_value()) {
            return result.error();
        }
        const float* in = inputs[0]->data<float>();
        float* out = result->data<float>();
        if (context.kernels != nullptr && context.kernels->softmax != nullptr) {
            context.kernels->softmax(in, out, rows, cols);
        } else {
            detail::softmax_reference(in, out, rows, cols);
        }
        *outputs[0] = std::move(*result);
        return {};
    }
};

}  // namespace ie
