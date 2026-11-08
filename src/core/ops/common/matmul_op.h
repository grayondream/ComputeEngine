#pragma once

#include <cstddef>
#include <cstdint>

#include "core/ops/op.h"
#include "core/runtime/thread_pool.h"

namespace ie {
namespace detail {

// Reference row kernel used when no capability-selected implementation is
// injected. Rows are independent, so threading cannot change the accumulation
// order of any output element.
inline void matmul_row_reference(const float* a_row, const float* b,
                                 float* out_row, std::int64_t k,
                                 std::int64_t n) {
    for (std::int64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (std::int64_t t = 0; t < k; ++t) {
            acc += a_row[t] * b[t * n + j];
        }
        out_row[j] = acc;
    }
}

}  // namespace detail

class MatMulOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        if (inputs.size() != 2 || outputs.empty() || outputs[0] == nullptr) {
            return Error::invalid_argument(
                "MatMul expects two inputs and one output");
        }
        if (context.allocator == nullptr) {
            return Error::invalid_argument("MatMul requires an allocator");
        }
        if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
            return valid.error();
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
                                       context.allocator);
        if (!result.has_value()) {
            return result.error();
        }
        const float* a = inputs[0]->data<float>();
        const float* b = inputs[1]->data<float>();
        float* out = result->data<float>();

        const KernelFns::MatMulRowFn row_fn =
            (context.kernels != nullptr &&
             context.kernels->matmul_row != nullptr)
                ? context.kernels->matmul_row
                : detail::matmul_row_reference;
        const auto compute_rows = [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const int64_t row = static_cast<int64_t>(i);
                row_fn(a + row * k, b, out + row * n, k, n);
            }
        };

        if (context.thread_pool != nullptr && m * n * k >= 32768) {
            context.thread_pool->parallel_for(static_cast<std::size_t>(m), 1,
                                              compute_rows);
        } else {
            compute_rows(0, static_cast<std::size_t>(m));
        }

        *outputs[0] = std::move(*result);
        return {};
    }
};

}  // namespace ie
