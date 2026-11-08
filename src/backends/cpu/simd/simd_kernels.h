#pragma once

#include <cstdint>

namespace ie::cpu {

// MatMul row kernel: out_row[j] = sum_t a_row[t] * b[t * n + j] for j in [0, n).
void matmul_row_scalar(const float* a_row, const float* b, float* out_row,
                       std::int64_t k, std::int64_t n);
void matmul_row_sse2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n);
void matmul_row_avx2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n);
void matmul_row_avx512(const float* a_row, const float* b, float* out_row,
                       std::int64_t k, std::int64_t n);

// Elementwise ReLU.
void relu_scalar(const float* in, float* out, std::int64_t n);
void relu_sse2(const float* in, float* out, std::int64_t n);
void relu_avx2(const float* in, float* out, std::int64_t n);
void relu_avx512(const float* in, float* out, std::int64_t n);

// Row-wise Softmax over a row-major [rows, cols] buffer. Subtracts the row max
// for numerical stability. Returns without touching memory when cols <= 0.
void softmax_scalar(const float* in, float* out, std::int64_t rows,
                    std::int64_t cols);
void softmax_avx2(const float* in, float* out, std::int64_t rows,
                  std::int64_t cols);
void softmax_avx512(const float* in, float* out, std::int64_t rows,
                    std::int64_t cols);

}  // namespace ie::cpu
