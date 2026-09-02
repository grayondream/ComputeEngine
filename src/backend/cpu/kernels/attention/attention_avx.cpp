#ifdef CE_HAS_AVX2
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace compute_engine {
namespace cpu {

static inline float HorizontalMax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_movehdup_ps(max128));
    return _mm_cvtss_f32(max128);
}

static inline float HorizontalSum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

void AttentionAVX(const float* Q, const float* K, const float* V, float* O,
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
                for (int k = 0; k < N; ++k) {
                    __m256 dot8 = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 qv = _mm256_loadu_ps(Q_bh + q * D + d);
                        __m256 kv = _mm256_loadu_ps(K_bh + k * D + d);
                        dot8 = _mm256_fmadd_ps(qv, kv, dot8);
                    }
                    float dot = HorizontalSum8(dot8);
                    for (; d < D; ++d)
                        dot += Q_bh[q * D + d] * K_bh[k * D + d];
                    scores[q * N + k] = dot * scale;
                }
            }

            for (int q = 0; q < N; ++q) {
                float max_val = scores[q * N];
                for (int k = 1; k < N; ++k)
                    max_val = std::max(max_val, scores[q * N + k]);

                __m256 max_v = _mm256_set1_ps(max_val);
                __m256 sum8 = _mm256_setzero_ps();
                int k = 0;
                for (; k + 8 <= N; k += 8) {
                    __m256 s = _mm256_loadu_ps(scores + q * N + k);
                    __m256 e = __builtin_ia32_exp_ps256(_mm256_sub_ps(s, max_v));
                    _mm256_storeu_ps(scores + q * N + k, e);
                    sum8 = _mm256_add_ps(sum8, e);
                }
                float sum = HorizontalSum8(sum8);
                for (; k < N; ++k) {
                    scores[q * N + k] = std::exp(scores[q * N + k] - max_val);
                    sum += scores[q * N + k];
                }
                float inv_sum = 1.0f / sum;
                __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
                k = 0;
                for (; k + 8 <= N; k += 8) {
                    __m256 s = _mm256_loadu_ps(scores + q * N + k);
                    _mm256_storeu_ps(scores + q * N + k, _mm256_mul_ps(s, inv_sum_v));
                }
                for (; k < N; ++k)
                    scores[q * N + k] *= inv_sum;
            }

            std::memset(O_bh, 0, N * D * sizeof(float));
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k < N; ++k) {
                    float attn = scores[q * N + k];
                    __m256 attn_v = _mm256_set1_ps(attn);
                    int d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 o = _mm256_loadu_ps(O_bh + q * D + d);
                        __m256 v = _mm256_loadu_ps(V_bh + k * D + d);
                        o = _mm256_fmadd_ps(attn_v, v, o);
                        _mm256_storeu_ps(O_bh + q * D + d, o);
                    }
                    for (; d < D; ++d)
                        O_bh[q * D + d] += attn * V_bh[k * D + d];
                }
            }
        }
    }

    std::free(scores);
}

void AttentionCausalAVX(const float* Q, const float* K, const float* V, float* O,
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
                int k = 0;
                for (; k + 8 <= q + 1; k += 8) {
                    __m256 dot8 = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 qv = _mm256_loadu_ps(Q_bh + q * D + d);
                        __m256 kv = _mm256_loadu_ps(K_bh + k * D + d);
                        dot8 = _mm256_fmadd_ps(qv, kv, dot8);
                    }
                    float dot_base = HorizontalSum8(dot8);
                    for (int kk = k; kk < std::min(k + 8, q + 1); ++kk) {
                        float dot = dot_base;
                        int dd = 0;
                        for (; dd + 8 <= D; dd += 8) {}
                        for (; dd < D; ++dd)
                            dot += Q_bh[q * D + dd] * K_bh[kk * D + dd];
                        scores[q * N + kk] = dot * scale;
                        max_val = std::max(max_val, dot * scale);
                    }
                }
                for (; k <= q; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += Q_bh[q * D + d] * K_bh[k * D + d];
                    scores[q * N + k] = dot * scale;
                    max_val = std::max(max_val, dot * scale);
                }
                for (int kk = q + 1; kk < N; ++kk)
                    scores[q * N + kk] = -1e30f;

                __m256 max_v = _mm256_set1_ps(max_val);
                __m256 sum8 = _mm256_setzero_ps();
                k = 0;
                for (; k + 8 <= N; k += 8) {
                    __m256 s = _mm256_loadu_ps(scores + q * N + k);
                    __m256 e = __builtin_ia32_exp_ps256(_mm256_sub_ps(s, max_v));
                    _mm256_storeu_ps(scores + q * N + k, e);
                    sum8 = _mm256_add_ps(sum8, e);
                }
                float sum = HorizontalSum8(sum8);
                for (; k < N; ++k) {
                    scores[q * N + k] = std::exp(scores[q * N + k] - max_val);
                    sum += scores[q * N + k];
                }
                float inv_sum = 1.0f / sum;
                __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
                k = 0;
                for (; k + 8 <= N; k += 8) {
                    __m256 s = _mm256_loadu_ps(scores + q * N + k);
                    _mm256_storeu_ps(scores + q * N + k, _mm256_mul_ps(s, inv_sum_v));
                }
                for (; k < N; ++k)
                    scores[q * N + k] *= inv_sum;
            }

            std::memset(O_bh, 0, N * D * sizeof(float));
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k <= q; ++k) {
                    float attn = scores[q * N + k];
                    __m256 attn_v = _mm256_set1_ps(attn);
                    int d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 o = _mm256_loadu_ps(O_bh + q * D + d);
                        __m256 v = _mm256_loadu_ps(V_bh + k * D + d);
                        o = _mm256_fmadd_ps(attn_v, v, o);
                        _mm256_storeu_ps(O_bh + q * D + d, o);
                    }
                    for (; d < D; ++d)
                        O_bh[q * D + d] += attn * V_bh[k * D + d];
                }
            }
        }
    }

    std::free(scores);
}

} // namespace cpu
} // namespace compute_engine
#endif // CE_HAS_AVX2
