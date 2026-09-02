#include <cmath>
#include <algorithm>
#include <cstring>

namespace compute_engine {
namespace cpu {

// Standard Scaled Dot-Product Attention (non-causal)
// Q, K, V layout: [B, H, N, D] (batch, heads, seq_len, head_dim)
// Output O: [B, H, N, D]
void AttentionScalar(const float* Q, const float* K, const float* V, float* O,
                     int B, int H, int N, int D) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    float* scores = static_cast<float*>(std::malloc(N * N * sizeof(float)));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* Q_bh = Q + (b * H + h) * N * D;
            const float* K_bh = K + (b * H + h) * N * D;
            const float* V_bh = V + (b * H + h) * N * D;
            float* O_bh = O + (b * H + h) * N * D;

            // scores = Q * K^T * scale
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k < N; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += Q_bh[q * D + d] * K_bh[k * D + d];
                    scores[q * N + k] = dot * scale;
                }
            }

            // softmax(scores)
            for (int q = 0; q < N; ++q) {
                float max_val = scores[q * N];
                for (int k = 1; k < N; ++k)
                    max_val = std::max(max_val, scores[q * N + k]);

                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    scores[q * N + k] = std::exp(scores[q * N + k] - max_val);
                    sum += scores[q * N + k];
                }
                float inv_sum = 1.0f / sum;
                for (int k = 0; k < N; ++k)
                    scores[q * N + k] *= inv_sum;
            }

            // O = scores * V
            std::memset(O_bh, 0, N * D * sizeof(float));
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k < N; ++k) {
                    float attn = scores[q * N + k];
                    for (int d = 0; d < D; ++d)
                        O_bh[q * D + d] += attn * V_bh[k * D + d];
                }
            }
        }
    }

    std::free(scores);
}

// Causal Scaled Dot-Product Attention
// Mask: scores[q][k] = -inf when k > q
void AttentionCausalScalar(const float* Q, const float* K, const float* V, float* O,
                           int B, int H, int N, int D) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    float* scores = static_cast<float*>(std::malloc(N * N * sizeof(float)));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* Q_bh = Q + (b * H + h) * N * D;
            const float* K_bh = K + (b * H + h) * N * D;
            const float* V_bh = V + (b * H + h) * N * D;
            float* O_bh = O + (b * H + h) * N * D;

            for (int q = 0; q < N; ++q) {
                float max_val = -1e30f;
                for (int k = 0; k <= q; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += Q_bh[q * D + d] * K_bh[k * D + d];
                    scores[q * N + k] = dot * scale;
                    max_val = std::max(max_val, dot * scale);
                }
                for (int k = q + 1; k < N; ++k)
                    scores[q * N + k] = -1e30f;

                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    scores[q * N + k] = std::exp(scores[q * N + k] - max_val);
                    sum += scores[q * N + k];
                }
                float inv_sum = 1.0f / sum;
                for (int k = 0; k < N; ++k)
                    scores[q * N + k] *= inv_sum;
            }

            std::memset(O_bh, 0, N * D * sizeof(float));
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k <= q; ++k) {
                    float attn = scores[q * N + k];
                    for (int d = 0; d < D; ++d)
                        O_bh[q * D + d] += attn * V_bh[k * D + d];
                }
            }
        }
    }

    std::free(scores);
}

} // namespace cpu
} // namespace compute_engine
