#ifdef CE_HAS_AVX2
#include <immintrin.h>

namespace compute_engine {
namespace cpu {

void ReLUAVX(const float* input, float* output, int size) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        _mm256_storeu_ps(output + i, _mm256_max_ps(v, zero));
    }
    for (; i < size; ++i) output[i] = std::max(0.0f, input[i]);
}

void SiLUAVX(const float* input, float* output, int size) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, __builtin_ia32_exp_ps256(neg_x)));
        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, sig));
    }
    for (; i < size; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void SigmoidAVX(const float* input, float* output, int size) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, __builtin_ia32_exp_ps256(neg_x)));
        _mm256_storeu_ps(output + i, sig);
    }
    for (; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

} // namespace cpu
} // namespace compute_engine
#endif
