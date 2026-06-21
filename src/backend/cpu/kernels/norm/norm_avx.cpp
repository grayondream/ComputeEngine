#ifdef CE_HAS_AVX2
#include <immintrin.h>

namespace compute_engine {
namespace cpu {

void LayerNormAVX(const float* input, const float* gamma, const float* beta,
                  float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        // Compute mean
        __m256 sum = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            sum = _mm256_add_ps(sum, v);
        }
        float mean = _mm256_cvtss_f32(_mm256_hadd_ps(sum, sum));
        mean += __builtin_shufflevector(_mm256_castps256_ps128(sum), _mm256_castps256_ps128(sum), 2, 3);
        mean /= cols;
        for (; c < cols; ++c) mean += row[c];
        mean /= cols;

        // Compute variance
        __m256 mean_v = _mm256_set1_ps(mean);
        __m256 var_sum = _mm256_setzero_ps();
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            __m256 d = _mm256_sub_ps(v, mean_v);
            var_sum = _mm256_fmadd_ps(d, d, var_sum);
        }
        float var = _mm256_cvtss_f32(_mm256_hadd_ps(var_sum, var_sum));
        var += __builtin_shufflevector(_mm256_castps256_ps128(var_sum), _mm256_castps256_ps128(var_sum), 2, 3);
        var /= cols;
        for (; c < cols; ++c) { float d = row[c] - mean; var += d * d; }
        var /= cols;

        float inv = 1.0f / __builtin_sqrtf(var + eps);
        __m256 inv_v = _mm256_set1_ps(inv);
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            __m256 g = _mm256_loadu_ps(gamma + c);
            __m256 b = _mm256_loadu_ps(beta + c);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(v, mean_v), inv_v);
            __m256 result = _mm256_fmadd_ps(g, norm, b);
            _mm256_storeu_ps(out + c, result);
        }
        for (; c < cols; ++c) {
            float norm = (row[c] - mean) * inv;
            out[c] = gamma[c] * norm + beta[c];
        }
    }
}

void GELUAVX(const float* input, float* output, int size) {
    const __m256 c = _mm256_set1_ps(0.7978845608f);
    const __m256 coeff = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
        __m256 t = _mm256_fmadd_ps(coeff, x3, x);
        t = _mm256_mul_ps(c, t);
        // Approximate tanh
        __m256 x2 = _mm256_mul_ps(t, t);
        __m256 p = _mm256_set1_ps(0.33333333f);
        p = _mm256_fmadd_ps(p, x2, one);
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t)));
        _mm256_storeu_ps(output + i, result);
    }
    for (; i < size; ++i) {
        float x = input[i];
        float t = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + t);
    }
}

void SoftMaxAVX(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = input + r * cols;
        float* out = output + r * cols;

        float max_val = row[0];
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            __m256 max8 = _mm256_max_ps(v, _mm256_permute2f128_ps(v, v, 1));
            max8 = _mm256_max_ps(max8, _mm256_shuffle_ps(max8, max8, 0x55));
            max_val = std::max(max_val, _mm256_cvtss_f32(max8));
        }
        for (; c < cols; ++c) max_val = std::max(max_val, row[c]);

        __m256 max_v = _mm256_set1_ps(max_val);
        __m256 sum = _mm256_setzero_ps();
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            __m256 e = __builtin_ia32_exp_ps256(_mm256_sub_ps(v, max_v));
            _mm256_storeu_ps(out + c, e);
            sum = _mm256_add_ps(sum, e);
        }
        float sum_f = _mm256_cvtss_f32(_mm256_hadd_ps(sum, sum));
        sum_f += __builtin_shufflevector(_mm256_castps256_ps128(sum), _mm256_castps256_ps128(sum), 2, 3);
        for (; c < cols; ++c) {
            out[c] = std::exp(row[c] - max_val);
            sum_f += out[c];
        }
        __m256 inv_sum = _mm256_set1_ps(1.0f / sum_f);
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 e = _mm256_loadu_ps(out + c);
            _mm256_storeu_ps(out + c, _mm256_mul_ps(e, inv_sum));
        }
        for (; c < cols; ++c) out[c] /= sum_f;
    }
}

} // namespace cpu
} // namespace compute_engine
#endif
