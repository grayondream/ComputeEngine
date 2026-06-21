// Reference Scalar MatMul implementation
#include "compute_engine/op/ref_ops.h"
#include <cmath>
#include <algorithm>

namespace compute_engine {
namespace reference {

void MatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void Conv2D(const float* input, const float* weight, float* output,
            int N, int C, int H, int W, int K, int R, int S,
            int pad_h, int pad_w, int stride_h, int stride_w) {
    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (W + 2 * pad_w - S) / stride_w + 1;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * stride_h - pad_h + r;
                                int iw = ow * stride_w - pad_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += input[((n * C + c) * H + ih) * W + iw]
                                         * weight[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    output[((n * K + k) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
}

void LayerNorm(const float* input, const float* gamma, const float* beta,
               float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += input[r * cols + c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = input[r * cols + c] - mean;
            var += d * d;
        }
        var /= cols;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            float x = (input[r * cols + c] - mean) * inv;
            output[r * cols + c] = gamma[c] * x + beta[c];
        }
    }
}

void RMSNorm(const float* input, const float* weight, float* output,
             int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c)
            sum_sq += input[r * cols + c] * input[r * cols + c];
        float rms = std::sqrt(sum_sq / cols + eps);
        for (int c = 0; c < cols; ++c)
            output[r * cols + c] = input[r * cols + c] / rms * weight[c];
    }
}

void BatchNorm(const float* input, const float* gamma, const float* beta,
               const float* mean, const float* variance,
               float* output, int N, int C, int H, int W, float eps) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    output[idx] = gamma[c] * (input[idx] - mean[c]) / std::sqrt(variance[c] + eps) + beta[c];
                }
}

void SoftMax(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float max_val = input[r * cols];
        for (int c = 1; c < cols; ++c)
            max_val = std::max(max_val, input[r * cols + c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] = std::exp(input[r * cols + c] - max_val);
            sum += output[r * cols + c];
        }
        for (int c = 0; c < cols; ++c)
            output[r * cols + c] /= sum;
    }
}

void GELU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float c = 0.7978845608f; // sqrt(2/pi)
        float t = std::tanh(c * (x + 0.044715f * x * x * x));
        output[i] = 0.5f * x * (1.0f + t);
    }
}

void SiLU(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void Attention(const float* Q, const float* K, const float* V, float* O,
               int B, int H, int N, int D) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int q = 0; q < N; ++q) {
                float max_val = -1e9f;
                for (int k = 0; k < N; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += Q[((b * H + h) * N + q) * D + d]
                             * K[((b * H + h) * N + k) * D + d];
                    }
                    max_val = std::max(max_val, dot * scale);
                }
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) O[((b * H + h) * N + q) * D + d] = 0.0f;
                for (int k = 0; k < N; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += Q[((b * H + h) * N + q) * D + d]
                             * K[((b * H + h) * N + k) * D + d];
                    }
                    float attn = std::exp((dot * scale) - max_val);
                    sum += attn;
                    for (int d = 0; d < D; ++d) {
                        O[((b * H + h) * N + q) * D + d] +=
                            attn * V[((b * H + h) * N + k) * D + d];
                    }
                }
                for (int d = 0; d < D; ++d)
                    O[((b * H + h) * N + q) * D + d] /= sum;
            }
        }
    }
}

void RoPE(float* Q, float* K, int B, int H, int N, int D,
          const float* sin_table, const float* cos_table) {
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int pos = 0; pos < N; ++pos) {
                float s = sin_table[pos], c = cos_table[pos];
                for (int d = 0; d < D; d += 2) {
                    float q0 = Q[((b * H + h) * N + pos) * D + d];
                    float q1 = Q[((b * H + h) * N + pos) * D + d + 1];
                    Q[((b * H + h) * N + pos) * D + d] = q0 * c - q1 * s;
                    Q[((b * H + h) * N + pos) * D + d + 1] = q0 * s + q1 * c;
                    float k0 = K[((b * H + h) * N + pos) * D + d];
                    float k1 = K[((b * H + h) * N + pos) * D + d + 1];
                    K[((b * H + h) * N + pos) * D + d] = k0 * c - k1 * s;
                    K[((b * H + h) * N + pos) * D + d + 1] = k0 * s + k1 * c;
                }
            }
}

void Add(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i) c[i] = a[i] + b[i];
}

void Mul(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i) c[i] = a[i] * b[i];
}

} // namespace reference
} // namespace compute_engine
