#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include "backends/cpu/simd/simd_kernels.h"

using namespace ie;

namespace {
bool has(std::string_view feature) {
#if defined(__x86_64__) || defined(__i386__)
    if (feature == "sse2") return __builtin_cpu_supports("sse2");
    if (feature == "avx2") return __builtin_cpu_supports("avx2");
    if (feature == "fma") return __builtin_cpu_supports("fma");
    if (feature == "avx512f") return __builtin_cpu_supports("avx512f");
    if (feature == "avx512bw") return __builtin_cpu_supports("avx512bw");
    if (feature == "avx512vl") return __builtin_cpu_supports("avx512vl");
    return false;
#else
    (void)feature;
    return false;
#endif
}
}  // namespace

TEST(SimdKernelsTest, MatMulVariantsMatchScalar) {
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
        {1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {64, 64, 64}};
    std::uint32_t state = 12345;
    auto next = [&] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xffff) / 65535.0f - 0.5f;
    };
    for (auto [m, k, n] : shapes) {
        std::vector<float> a(m * k), b(k * n), expected(m * n), actual(m * n);
        for (float& v : a) v = next();
        for (float& v : b) v = next();
        for (int64_t i = 0; i < m; ++i) {
            cpu::matmul_row_scalar(a.data() + i * k, b.data(), expected.data() + i * n, k, n);
        }
        if (has("sse2")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_sse2(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
        if (has("avx2") && has("fma")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_avx2(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
        if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_avx512(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
    }
}

TEST(SimdKernelsTest, ReluAndSoftmaxVariantsMatchScalar) {
    std::vector<float> in = {-3.0f, -0.0f, 0.0f, 2.0f, 1000.0f, -1000.0f, 7.0f};
    std::vector<float> relu_expected(in.size()), relu_actual(in.size());
    cpu::relu_scalar(in.data(), relu_expected.data(), in.size());
    if (has("avx2") && has("fma")) {
        cpu::relu_avx2(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        cpu::relu_avx512(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }

    std::vector<float> softmax_expected(in.size()), softmax_actual(in.size());
    cpu::softmax_scalar(in.data(), softmax_expected.data(), 1, in.size());
    if (has("avx2") && has("fma")) {
        cpu::softmax_avx2(in.data(), softmax_actual.data(), 1, in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            EXPECT_NEAR(softmax_actual[i], softmax_expected[i], 1e-6f);
        }
    }
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        cpu::softmax_avx512(in.data(), softmax_actual.data(), 1, in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            EXPECT_NEAR(softmax_actual[i], softmax_expected[i], 1e-6f);
        }
    }
}
