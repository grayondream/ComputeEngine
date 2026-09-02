#ifdef CE_HAS_NEON
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

namespace compute_engine {
namespace cpu {

void ReLUNeon(const float* input, float* output, int size) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        vst1q_f32(output + i, vmaxq_f32(v, zero));
    }
    for (; i < size; ++i)
        output[i] = std::max(0.0f, input[i]);
}

void LeakyReLUNeon(const float* input, float* output, int size, float alpha) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float32x4_t neg = vmulq_f32(v, alpha_v);
        uint32x4_t mask = vcgeq_f32(v, zero);
        float32x4_t result = vbslq_f32(mask, v, neg);
        vst1q_f32(output + i, result);
    }
    for (; i < size; ++i)
        output[i] = input[i] >= 0.0f ? input[i] : alpha * input[i];
}

void GELUNeon(const float* input, float* output, int size) {
    const float32x4_t c = vdupq_n_f32(0.7978845608f);
    const float32x4_t coeff = vdupq_n_f32(0.044715f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t x3 = vmulq_f32(x, vmulq_f32(x, x));
        float32x4_t t = vmlaq_f32(x, coeff, x3);
        t = vmulq_f32(c, t);
        float32x4_t result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, vtanhq_f32(t))));
        vst1q_f32(output + i, result);
    }
    for (; i < size; ++i) {
        float x = input[i];
        float t = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
}

void SiLUNeon(const float* input, float* output, int size) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, vexpq_f32(neg_x)));
        vst1q_f32(output + i, vmulq_f32(x, sig));
    }
    for (; i < size; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void SigmoidNeon(const float* input, float* output, int size) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, vexpq_f32(neg_x)));
        vst1q_f32(output + i, sig);
    }
    for (; i < size; ++i)
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
}

void TanhNeon(const float* input, float* output, int size) {
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        vst1q_f32(output + i, vtanhq_f32(x));
    }
    for (; i < size; ++i)
        output[i] = std::tanh(input[i]);
}

} // namespace cpu
} // namespace compute_engine
#endif // CE_HAS_NEON
