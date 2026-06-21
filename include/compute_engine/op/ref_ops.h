#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/tensor.h"
#include <cstdint>

namespace compute_engine {

namespace reference {

void MatMul(const float* A, const float* B, float* C, int M, int N, int K);

void Conv2D(const float* input, const float* weight, float* output,
            int N, int C, int H, int W, int K, int R, int S,
            int pad_h, int pad_w, int stride_h, int stride_w);

void LayerNorm(const float* input, const float* gamma, const float* beta,
               float* output, int rows, int cols, float eps);

void RMSNorm(const float* input, const float* weight, float* output,
             int rows, int cols, float eps);

void BatchNorm(const float* input, const float* gamma, const float* beta,
               const float* mean, const float* variance,
               float* output, int N, int C, int H, int W, float eps);

void SoftMax(const float* input, float* output, int rows, int cols);

void GELU(const float* input, float* output, int size);

void SiLU(const float* input, float* output, int size);

void Attention(const float* Q, const float* K, const float* V, float* O,
               int B, int H, int N, int D);

void RoPE(float* Q, float* K, int B, int H, int N, int D,
          const float* sin_table, const float* cos_table);

void Add(const float* a, const float* b, float* c, int size);

void Mul(const float* a, const float* b, float* c, int size);

} // namespace reference
} // namespace compute_engine
