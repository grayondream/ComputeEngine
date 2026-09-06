#include "compute_engine/backend/cpu_dispatch.h"
#include "compute_engine/platform/logger.h"
#include <cstring>

#ifdef CE_HAS_AVX2
#include <cpuid.h>
#endif

namespace compute_engine {

static CPUCaps g_caps{};
static bool g_caps_initialized = false;

static void DetectCPUFeatures() {
#ifdef CE_HAS_AVX2
    uint32_t eax, ebx, ecx, edx;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    bool has_avx2     = (ebx >> 5) & 1;
    bool has_avx512f  = (ebx >> 16) & 1;
    bool has_avx512bw = (ebx >> 30) & 1;

    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    bool has_sse42 = (ecx >> 20) & 1;
    bool has_avx   = (ecx >> 28) & 1;
    bool has_fma   = (ecx >> 12) & 1;

    g_caps.sse42    = has_sse42;
    g_caps.avx      = has_avx;
    g_caps.avx2     = has_avx2;
    g_caps.fma      = has_fma;
    g_caps.avx512f  = has_avx512f;
    g_caps.avx512bw = has_avx512bw;

    LOGI("CPUID: SSE4.2={} AVX={} AVX2={} FMA={} AVX512F={} AVX512BW={}",
         has_sse42, has_avx, has_avx2, has_fma, has_avx512f, has_avx512bw);
#else
    LOGI("CPUID: AVX detection not enabled at compile time");
#endif
    g_caps_initialized = true;
}

const CPUCaps& GetCPUCaps() {
    if (!g_caps_initialized) DetectCPUFeatures();
    return g_caps;
}

} // namespace compute_engine

// ══════════════════════════════════════════════════════════════════════════════
// Kernel forward declarations (conditionally compiled)
// ══════════════════════════════════════════════════════════════════════════════

namespace compute_engine {
namespace cpu {

// ─── MatMul ───
void MatMulScalar(const float*, const float*, float*, int, int, int);
void MatMulScalarTiled(const float*, const float*, float*, int, int, int, int);

#ifdef CE_HAS_AVX2
void MatMulAVX(const float*, const float*, float*, int, int, int);
#endif
#ifdef CE_HAS_AVX512
void MatMulAVX512(const float*, const float*, float*, int, int, int);
#endif

// ─── Conv2D ───
void Conv2DScalar(const float*, const float*, float*,
                  int, int, int, int, int, int, int, int, int, int, int);
#ifdef CE_HAS_AVX2
void Conv2DAVX(const float*, const float*, float*,
               int, int, int, int, int, int, int, int, int, int, int);
#endif

// ─── Norm ───
void LayerNormScalar(const float*, const float*, const float*,
                     float*, int, int, float);
void RMSNormScalar(const float*, const float*, float*, int, int, float);
void BatchNormScalar(const float*, const float*, const float*,
                     const float*, const float*, float*, int, int, int, int, float);
void SoftMaxScalar(const float*, float*, int, int);

#ifdef CE_HAS_AVX2
void LayerNormAVX(const float*, const float*, const float*,
                  float*, int, int, float);
void SoftMaxAVX(const float*, float*, int, int);
#endif
#ifdef CE_HAS_NEON
void LayerNormNEON(const float*, const float*, const float*,
                   float*, int, int, float);
void RMSNormNEON(const float*, const float*, float*, int, int, float);
void BatchNormNEON(const float*, const float*, const float*,
                   const float*, const float*, float*, int, int, int, int, float);
void SoftMaxNEON(const float*, float*, int, int);
#endif

// ─── Activation ───
void ReLUScalar(const float*, float*, int);
void LeakyReLUScalar(const float*, float*, int, float);
void GELUScalar(const float*, float*, int);
void SiLUScalar(const float*, float*, int);
void SigmoidScalar(const float*, float*, int);
void TanhScalar(const float*, float*, int);

#ifdef CE_HAS_AVX2
void ReLUAVX(const float*, float*, int);
void SiLUAVX(const float*, float*, int);
void SigmoidAVX(const float*, float*, int);
void GELUAVX(const float*, float*, int);
#endif
#ifdef CE_HAS_NEON
void ReLUNeon(const float*, float*, int);
void LeakyReLUNeon(const float*, float*, int, float);
void GELUNeon(const float*, float*, int);
void SiLUNeon(const float*, float*, int);
void SigmoidNeon(const float*, float*, int);
void TanhNeon(const float*, float*, int);
#endif

// ─── Attention ───
void AttentionScalar(const float*, const float*, const float*, float*,
                     int, int, int, int);
void AttentionCausalScalar(const float*, const float*, const float*, float*,
                           int, int, int, int);
#ifdef CE_HAS_AVX2
void AttentionAVX(const float*, const float*, const float*, float*,
                  int, int, int, int);
void AttentionCausalAVX(const float*, const float*, const float*, float*,
                        int, int, int, int);
#endif

} // namespace cpu
} // namespace compute_engine

// ══════════════════════════════════════════════════════════════════════════════
// Dispatch implementations
// ══════════════════════════════════════════════════════════════════════════════

namespace compute_engine {
namespace cpu {

void MatMulDispatch(const float* A, const float* B, float* C, int M, int N, int K) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX512
    if (caps.avx512f) { MatMulAVX512(A, B, C, M, N, K); return; }
#endif
#ifdef CE_HAS_AVX2
    if (caps.avx2) { MatMulAVX(A, B, C, M, N, K); return; }
#endif
    MatMulScalarTiled(A, B, C, M, N, K, 32);
}

void Conv2DDispatch(const float* input, const float* weight, float* output,
                    int N, int C, int H, int W,
                    int K, int R, int S,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) {
        Conv2DAVX(input, weight, output, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
        return;
    }
#endif
    Conv2DScalar(input, weight, output, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
}

void LayerNormDispatch(const float* input, const float* gamma, const float* beta,
                       float* output, int rows, int cols, float eps) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) { LayerNormNEON(input, gamma, beta, output, rows, cols, eps); return; }
    // Note: on x86, NEON won't be compiled, so AVX path below handles it
#endif
#ifdef CE_HAS_AVX2
    if (caps.avx2) { LayerNormAVX(input, gamma, beta, output, rows, cols, eps); return; }
#endif
    LayerNormScalar(input, gamma, beta, output, rows, cols, eps);
}

void RMSNormDispatch(const float* input, const float* weight, float* output,
                     int rows, int cols, float eps) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) { RMSNormNEON(input, weight, output, rows, cols, eps); return; }
#endif
    RMSNormScalar(input, weight, output, rows, cols, eps);
}

void BatchNormDispatch(const float* input, const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       float* output, int N, int C, int H, int W, float eps) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) {
        BatchNormNEON(input, gamma, beta, mean, variance, output, N, C, H, W, eps);
        return;
    }
#endif
    BatchNormScalar(input, gamma, beta, mean, variance, output, N, C, H, W, eps);
}

void SoftMaxDispatch(const float* input, float* output, int rows, int cols) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) { SoftMaxNEON(input, output, rows, cols); return; }
#endif
#ifdef CE_HAS_AVX2
    if (caps.avx2) { SoftMaxAVX(input, output, rows, cols); return; }
#endif
    SoftMaxScalar(input, output, rows, cols);
}

void ReLUDispatch(const float* input, float* output, int size) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) { ReLUAVX(input, output, size); return; }
#endif
#ifdef CE_HAS_NEON
    if (caps.sse42) { ReLUNeon(input, output, size); return; }
#endif
    ReLUScalar(input, output, size);
}

void LeakyReLUDispatch(const float* input, float* output, int size, float alpha) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) { LeakyReLUNeon(input, output, size, alpha); return; }
#endif
    LeakyReLUScalar(input, output, size, alpha);
}

void GELUDispatch(const float* input, float* output, int size) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) { GELUAVX(input, output, size); return; }
#endif
#ifdef CE_HAS_NEON
    if (caps.sse42) { GELUNeon(input, output, size); return; }
#endif
    GELUScalar(input, output, size);
}

void SiLUDispatch(const float* input, float* output, int size) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) { SiLUAVX(input, output, size); return; }
#endif
#ifdef CE_HAS_NEON
    if (caps.sse42) { SiLUNeon(input, output, size); return; }
#endif
    SiLUScalar(input, output, size);
}

void SigmoidDispatch(const float* input, float* output, int size) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) { SigmoidAVX(input, output, size); return; }
#endif
#ifdef CE_HAS_NEON
    if (caps.sse42) { SigmoidNeon(input, output, size); return; }
#endif
    SigmoidScalar(input, output, size);
}

void TanhDispatch(const float* input, float* output, int size) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_NEON
    if (caps.sse42) { TanhNeon(input, output, size); return; }
#endif
    TanhScalar(input, output, size);
}

void AttentionDispatch(const float* Q, const float* K, const float* V, float* O,
                       int B, int H, int N, int D, bool causal) {
    const auto& caps = GetCPUCaps();
#ifdef CE_HAS_AVX2
    if (caps.avx2) {
        if (causal) AttentionCausalAVX(Q, K, V, O, B, H, N, D);
        else        AttentionAVX(Q, K, V, O, B, H, N, D);
        return;
    }
#endif
    if (causal) AttentionCausalScalar(Q, K, V, O, B, H, N, D);
    else        AttentionScalar(Q, K, V, O, B, H, N, D);
}

} // namespace cpu
} // namespace compute_engine
