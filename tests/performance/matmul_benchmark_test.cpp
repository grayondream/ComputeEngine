#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <vector>

#include "backends/cpu/simd/simd_kernels.h"

using namespace ie;

namespace {
bool has_avx2_fma() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool has_avx512() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}
}  // namespace

TEST(MatMulBenchmarkSanity, Avx2BeatsScalarOnLargeMatrix) {
#ifndef NDEBUG
    GTEST_SKIP() << "performance sanity runs in Release";
#endif
    if (!has_avx2_fma()) {
        GTEST_SKIP() << "AVX2+FMA not available";
    }
    constexpr int64_t m = 256, k = 256, n = 256;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), out(m * n);
    auto run = [&](auto fn) {
        auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 3; ++rep) {
            for (int64_t i = 0; i < m; ++i) {
                fn(a.data() + i * k, b.data(), out.data() + i * n, k, n);
            }
        }
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
    const double scalar = run(cpu::matmul_row_scalar);
    const double avx2 = run(cpu::matmul_row_avx2);
    EXPECT_LT(avx2, scalar * 0.85) << "scalar=" << scalar << " avx2=" << avx2;
    if (has_avx512()) {
        const double avx512 = run(cpu::matmul_row_avx512);
        EXPECT_LT(avx512, scalar * 0.85)
            << "scalar=" << scalar << " avx512=" << avx512;
    }
}
