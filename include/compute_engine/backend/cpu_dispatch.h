#pragma once

#include "compute_engine/core/macros.h"
#include <cstdint>

namespace compute_engine {

struct CPUCaps {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
};

const CPUCaps& GetCPUCaps();

namespace cpu {

// ─── MatMul ───
void MatMulDispatch(const float* A, const float* B, float* C, int M, int N, int K);

// ─── Conv2D ───
void Conv2DDispatch(const float* input, const float* weight, float* output,
                    int N, int C, int H, int W,
                    int K, int R, int S,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w);

// ─── Normalization ───
void LayerNormDispatch(const float* input, const float* gamma, const float* beta,
                       float* output, int rows, int cols, float eps);

void RMSNormDispatch(const float* input, const float* weight, float* output,
                     int rows, int cols, float eps);

void BatchNormDispatch(const float* input, const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       float* output, int N, int C, int H, int W, float eps);

void SoftMaxDispatch(const float* input, float* output, int rows, int cols);

// ─── Activation ───
void ReLUDispatch(const float* input, float* output, int size);

void LeakyReLUDispatch(const float* input, float* output, int size, float alpha);

void GELUDispatch(const float* input, float* output, int size);

void SiLUDispatch(const float* input, float* output, int size);

void SigmoidDispatch(const float* input, float* output, int size);

void TanhDispatch(const float* input, float* output, int size);

// ─── Attention ───
void AttentionDispatch(const float* Q, const float* K, const float* V, float* O,
                       int B, int H, int N, int D, bool causal = false);

} // namespace cpu
} // namespace compute_engine
