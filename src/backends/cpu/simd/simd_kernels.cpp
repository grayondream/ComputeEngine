#include "backends/cpu/simd/simd_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// Function-level ISA targeting keeps the rest of the translation unit at the
// project baseline: no global -march/-mavx flags are used. Non-x86 and
// non-GNU/Clang builds get scalar-forwarding placeholders instead; the
// capability registry never selects them there because the caps stay false.
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define IE_CPU_SIMD_X86 1
#define IE_CPU_TARGET(...) __attribute__((target(__VA_ARGS__)))
#else
#define IE_CPU_SIMD_X86 0
#define IE_CPU_TARGET(...)
#endif

namespace ie::cpu {
namespace {

void matmul_row_tail_scalar(const float* a_row, const float* b, float* out_row,
                            std::int64_t k, std::int64_t n,
                            std::int64_t begin) {
    for (std::int64_t j = begin; j < n; ++j) {
        float acc = 0.0f;
        for (std::int64_t t = 0; t < k; ++t) {
            acc += a_row[t] * b[t * n + j];
        }
        out_row[j] = acc;
    }
}

void softmax_row_scalar(const float* row, float* dst, std::int64_t cols) {
    float max_value = row[0];
    for (std::int64_t i = 1; i < cols; ++i) {
        max_value = std::max(max_value, row[i]);
    }
    float sum = 0.0f;
    for (std::int64_t i = 0; i < cols; ++i) {
        dst[i] = std::exp(row[i] - max_value);
        sum += dst[i];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (std::int64_t i = 0; i < cols; ++i) {
        dst[i] *= inv;
    }
}

}  // namespace

void matmul_row_scalar(const float* a_row, const float* b, float* out_row,
                       std::int64_t k, std::int64_t n) {
    if (out_row == nullptr || n <= 0) {
        return;
    }
    matmul_row_tail_scalar(a_row, b, out_row, k, n, 0);
}

void relu_scalar(const float* in, float* out, std::int64_t n) {
    if (in == nullptr || out == nullptr || n <= 0) {
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void softmax_scalar(const float* in, float* out, std::int64_t rows,
                    std::int64_t cols) {
    if (in == nullptr || out == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    for (std::int64_t r = 0; r < rows; ++r) {
        softmax_row_scalar(in + r * cols, out + r * cols, cols);
    }
}

#if IE_CPU_SIMD_X86

IE_CPU_TARGET("sse2")
void matmul_row_sse2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n) {
    if (out_row == nullptr || n <= 0) {
        return;
    }
    std::int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
        __m128 acc = _mm_setzero_ps();
        for (std::int64_t t = 0; t < k; ++t) {
            const __m128 a = _mm_set1_ps(a_row[t]);
            const __m128 bv = _mm_loadu_ps(b + t * n + j);
            acc = _mm_add_ps(acc, _mm_mul_ps(a, bv));
        }
        _mm_storeu_ps(out_row + j, acc);
    }
    matmul_row_tail_scalar(a_row, b, out_row, k, n, j);
}

IE_CPU_TARGET("avx2,fma")
void matmul_row_avx2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n) {
    if (out_row == nullptr || n <= 0) {
        return;
    }
    std::int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (std::int64_t t = 0; t < k; ++t) {
            const __m256 a = _mm256_set1_ps(a_row[t]);
            const __m256 bv = _mm256_loadu_ps(b + t * n + j);
            acc = _mm256_fmadd_ps(a, bv, acc);
        }
        _mm256_storeu_ps(out_row + j, acc);
    }
    matmul_row_tail_scalar(a_row, b, out_row, k, n, j);
}

IE_CPU_TARGET("avx512f,avx512bw,avx512vl")
void matmul_row_avx512(const float* a_row, const float* b, float* out_row,
                       std::int64_t k, std::int64_t n) {
    if (out_row == nullptr || n <= 0) {
        return;
    }
    std::int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        __m512 acc = _mm512_setzero_ps();
        for (std::int64_t t = 0; t < k; ++t) {
            const __m512 a = _mm512_set1_ps(a_row[t]);
            const __m512 bv = _mm512_loadu_ps(b + t * n + j);
            acc = _mm512_fmadd_ps(a, bv, acc);
        }
        _mm512_storeu_ps(out_row + j, acc);
    }
    matmul_row_tail_scalar(a_row, b, out_row, k, n, j);
}

IE_CPU_TARGET("sse2")
void relu_sse2(const float* in, float* out, std::int64_t n) {
    if (in == nullptr || out == nullptr || n <= 0) {
        return;
    }
    const __m128 zero = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 v = _mm_loadu_ps(in + i);
        _mm_storeu_ps(out + i, _mm_max_ps(v, zero));
    }
    for (; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

IE_CPU_TARGET("avx2,fma")
void relu_avx2(const float* in, float* out, std::int64_t n) {
    if (in == nullptr || out == nullptr || n <= 0) {
        return;
    }
    const __m256 zero = _mm256_setzero_ps();
    std::int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

IE_CPU_TARGET("avx512f,avx512bw,avx512vl")
void relu_avx512(const float* in, float* out, std::int64_t n) {
    if (in == nullptr || out == nullptr || n <= 0) {
        return;
    }
    const __m512 zero = _mm512_setzero_ps();
    std::int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_max_ps(v, zero));
    }
    for (; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

IE_CPU_TARGET("avx2,fma")
void softmax_avx2(const float* in, float* out, std::int64_t rows,
                  std::int64_t cols) {
    if (in == nullptr || out == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = in + r * cols;
        float* dst = out + r * cols;

        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        std::int64_t i = 0;
        for (; i + 8 <= cols; i += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + i));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, vmax);
        float max_value = lanes[0];
        for (int lane = 1; lane < 8; ++lane) {
            max_value = std::max(max_value, lanes[lane]);
        }
        for (; i < cols; ++i) {
            max_value = std::max(max_value, row[i]);
        }

        float sum = 0.0f;
        for (i = 0; i < cols; ++i) {
            dst[i] = std::exp(row[i] - max_value);
            sum += dst[i];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 8 <= cols; i += 8) {
            const __m256 v = _mm256_loadu_ps(dst + i);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, vinv));
        }
        for (; i < cols; ++i) {
            dst[i] *= inv;
        }
    }
}

IE_CPU_TARGET("avx512f,avx512bw,avx512vl")
void softmax_avx512(const float* in, float* out, std::int64_t rows,
                    std::int64_t cols) {
    if (in == nullptr || out == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = in + r * cols;
        float* dst = out + r * cols;

        __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        std::int64_t i = 0;
        for (; i + 16 <= cols; i += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + i));
        }
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, vmax);
        float max_value = lanes[0];
        for (int lane = 1; lane < 16; ++lane) {
            max_value = std::max(max_value, lanes[lane]);
        }
        for (; i < cols; ++i) {
            max_value = std::max(max_value, row[i]);
        }

        float sum = 0.0f;
        for (i = 0; i < cols; ++i) {
            dst[i] = std::exp(row[i] - max_value);
            sum += dst[i];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const __m512 vinv = _mm512_set1_ps(inv);
        i = 0;
        for (; i + 16 <= cols; i += 16) {
            const __m512 v = _mm512_loadu_ps(dst + i);
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(v, vinv));
        }
        for (; i < cols; ++i) {
            dst[i] *= inv;
        }
    }
}

#else  // !IE_CPU_SIMD_X86

// Cross-platform placeholders: safe, never selected because the registry
// requires caps that a non-x86 profile cannot advertise.
void matmul_row_sse2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n) {
    matmul_row_scalar(a_row, b, out_row, k, n);
}

void matmul_row_avx2(const float* a_row, const float* b, float* out_row,
                     std::int64_t k, std::int64_t n) {
    matmul_row_scalar(a_row, b, out_row, k, n);
}

void matmul_row_avx512(const float* a_row, const float* b, float* out_row,
                       std::int64_t k, std::int64_t n) {
    matmul_row_scalar(a_row, b, out_row, k, n);
}

void relu_sse2(const float* in, float* out, std::int64_t n) {
    relu_scalar(in, out, n);
}

void relu_avx2(const float* in, float* out, std::int64_t n) {
    relu_scalar(in, out, n);
}

void relu_avx512(const float* in, float* out, std::int64_t n) {
    relu_scalar(in, out, n);
}

void softmax_avx2(const float* in, float* out, std::int64_t rows,
                  std::int64_t cols) {
    softmax_scalar(in, out, rows, cols);
}

void softmax_avx512(const float* in, float* out, std::int64_t rows,
                    std::int64_t cols) {
    softmax_scalar(in, out, rows, cols);
}

#endif  // IE_CPU_SIMD_X86

}  // namespace ie::cpu

#undef IE_CPU_SIMD_X86
#undef IE_CPU_TARGET
