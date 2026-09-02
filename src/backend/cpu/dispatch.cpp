#include "compute_engine/backend/cpu_backend.h"
#include "compute_engine/platform/logger.h"
#include <cstdint>
#include <cstring>

#ifdef CE_HAS_AVX2
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace compute_engine {

struct CPUCaps {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
};

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
