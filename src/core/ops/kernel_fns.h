#pragma once

#include <cstdint>

namespace ie {

// Concrete F32 kernel entry points selected at runtime by the CPU catalog.
// A null member means the catalog has no implementation for that op.
struct KernelFns {
    using MatMulRowFn = void (*)(const float* a_row, const float* b,
                                 float* out_row, std::int64_t k, std::int64_t n);
    using UnaryFn = void (*)(const float* in, float* out, std::int64_t n);
    using SoftmaxFn = void (*)(const float* in, float* out,
                               std::int64_t rows, std::int64_t cols);

    MatMulRowFn matmul_row{nullptr};
    UnaryFn relu{nullptr};
    SoftmaxFn softmax{nullptr};
};

}  // namespace ie
