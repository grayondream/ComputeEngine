#ifdef CE_HAS_NEON
#include <arm_neon.h>
#include <cmath>

namespace compute_engine {
namespace cpu {

void LayerNormNEON(const float* input, const float* gamma, const float* beta,
                   float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float32x4_t sum = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            sum = vaddq_f32(sum, v);
        }
        float mean = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1)
                   + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        for (; c < cols; ++c) mean += row[c];
        mean /= cols;

        float32x4_t mean_v = vdupq_n_f32(mean);
        float32x4_t var_sum = vdupq_n_f32(0.0f);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            float32x4_t d = vsubq_f32(v, mean_v);
            var_sum = vmlaq_f32(var_sum, d, d);
        }
        float var = vgetq_lane_f32(var_sum, 0) + vgetq_lane_f32(var_sum, 1)
                  + vgetq_lane_f32(var_sum, 2) + vgetq_lane_f32(var_sum, 3);
        for (; c < cols; ++c) { float d = row[c] - mean; var += d * d; }
        var /= cols;

        float inv = 1.0f / std::sqrt(var + eps);
        float32x4_t inv_v = vdupq_n_f32(inv);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            float32x4_t g = vld1q_f32(gamma + c);
            float32x4_t b = vld1q_f32(beta + c);
            float32x4_t norm = vmulq_f32(vsubq_f32(v, mean_v), inv_v);
            float32x4_t result = vmlaq_f32(b, g, norm);
            vst1q_f32(out + c, result);
        }
        for (; c < cols; ++c) {
            float norm = (row[c] - mean) * inv;
            out[c] = gamma[c] * norm + beta[c];
        }
    }
}

void RMSNormNEON(const float* input, const float* weight, float* output,
                 int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float32x4_t sum_sq = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            sum_sq = vmlaq_f32(sum_sq, v, v);
        }
        float sum = vgetq_lane_f32(sum_sq, 0) + vgetq_lane_f32(sum_sq, 1)
                  + vgetq_lane_f32(sum_sq, 2) + vgetq_lane_f32(sum_sq, 3);
        for (; c < cols; ++c) sum += row[c] * row[c];

        float rms = std::sqrt(sum / cols + eps);
        float inv_rms = 1.0f / rms;
        float32x4_t inv_v = vdupq_n_f32(inv_rms);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            float32x4_t w = vld1q_f32(weight + c);
            float32x4_t result = vmulq_f32(vmulq_f32(v, inv_v), w);
            vst1q_f32(out + c, result);
        }
        for (; c < cols; ++c)
            out[c] = row[c] * inv_rms * weight[c];
    }
}

void BatchNormNEON(const float* input, const float* gamma, const float* beta,
                   const float* mean, const float* variance,
                   float* output, int N, int C, int H, int W, float eps) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            float inv = 1.0f / std::sqrt(variance[c] + eps);
            float scale = gamma[c] * inv;
            float shift = -mean[c] * scale + beta[c];
            float32x4_t scale_v = vdupq_n_f32(scale);
            float32x4_t shift_v = vdupq_n_f32(shift);
            int hw = H * W;
            int idx = n * C * H * W + c * H * W;
            int i = 0;
            for (; i + 4 <= hw; i += 4) {
                float32x4_t v = vld1q_f32(input + idx + i);
                float32x4_t result = vmlaq_f32(shift_v, v, scale_v);
                vst1q_f32(output + idx + i, result);
            }
            for (; i < hw; ++i)
                output[idx + i] = input[idx + i] * scale + shift;
        }
}

void SoftMaxNEON(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float max_val = row[0];
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            float32x4_t max4 = vmaxq_f32(v, vrev64q_f32(v));
            max4 = vmaxq_f32(max4, vextq_f32(max4, max4, 2));
            max_val = std::max(max_val, vgetq_lane_f32(max4, 0));
        }
        for (; c < cols; ++c) max_val = std::max(max_val, row[c]);

        float32x4_t max_v = vdupq_n_f32(max_val);
        float32x4_t sum = vdupq_n_f32(0.0f);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vld1q_f32(row + c);
            float32x4_t e = expvq_f32(vsubq_f32(v, max_v));
            vst1q_f32(out + c, e);
            sum = vaddq_f32(sum, e);
        }
        float sum_f = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1)
                    + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        for (; c < cols; ++c) {
            out[c] = std::exp(row[c] - max_val);
            sum_f += out[c];
        }
        float32x4_t inv_sum = vdupq_n_f32(1.0f / sum_f);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t e = vld1q_f32(out + c);
            vst1q_f32(out + c, vmulq_f32(e, inv_sum));
        }
        for (; c < cols; ++c) out[c] /= sum_f;
    }
}

} // namespace cpu
} // namespace compute_engine
#endif // CE_HAS_NEON
