#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

std::vector<float> make_relu_input(std::int64_t n) {
    std::vector<float> in(static_cast<std::size_t>(n));
    std::uint32_t state = 0x9e3779b9u;
    for (float& v : in) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>((state >> 8) & 0xffff) / 65535.0f * 20.0f - 10.0f;
    }
    if (n > 0) in[0] = -0.0f;
    if (n > 1) in[1] = 0.0f;
    if (n > 2) in[2] = -1000.0f;
    if (n > 3) in[n - 1] = 1000.0f;
    return in;
}

// Exercises every vector width and tail by comparing against the scalar kernel.
void expect_relu_matches_scalar(std::int64_t n) {
    const std::vector<float> in = make_relu_input(n);
    std::vector<float> expected(in.size()), actual(in.size(), -1234.0f);
    cpu::relu_scalar(in.data(), expected.data(), n);
    auto check = [&](auto fn, const char* name) {
        std::fill(actual.begin(), actual.end(), -1234.0f);
        fn(in.data(), actual.data(), n);
        EXPECT_EQ(actual, expected) << name << " n=" << n;
    };
    if (has("sse2")) check(cpu::relu_sse2, "relu_sse2");
    if (has("avx2") && has("fma")) check(cpu::relu_avx2, "relu_avx2");
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        check(cpu::relu_avx512, "relu_avx512");
    }
}

std::vector<float> make_softmax_input(std::int64_t rows, std::int64_t cols) {
    std::vector<float> in(static_cast<std::size_t>(rows * cols));
    std::uint32_t state = 0x12345678u;
    for (float& v : in) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>((state >> 8) & 0xffff) / 65535.0f * 8.0f - 4.0f;
    }
    // Place each row's maximum away from index 0 and make it large: the
    // subtraction of the row max is required to avoid overflow, and vector
    // reductions must see it.
    for (std::int64_t r = 0; r < rows; ++r) {
        in[static_cast<std::size_t>(r * cols + cols / 2)] =
            (r % 2 == 0) ? 1000.0f : 700.0f;
    }
    return in;
}

void expect_softmax_rows_normalized(const std::vector<float>& out,
                                    std::int64_t rows, std::int64_t cols) {
    for (std::int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::int64_t c = 0; c < cols; ++c) {
            const float v = out[static_cast<std::size_t>(r * cols + c)];
            EXPECT_TRUE(std::isfinite(v)) << "row=" << r << " col=" << c;
            EXPECT_GE(v, 0.0f) << "row=" << r << " col=" << c;
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "row=" << r << " cols=" << cols;
    }
}

void expect_softmax_matches_scalar(std::int64_t rows, std::int64_t cols) {
    const std::vector<float> in = make_softmax_input(rows, cols);
    std::vector<float> expected(in.size()), actual(in.size(), -1234.0f);
    cpu::softmax_scalar(in.data(), expected.data(), rows, cols);
    expect_softmax_rows_normalized(expected, rows, cols);

    auto check = [&](auto fn, const char* name) {
        std::fill(actual.begin(), actual.end(), -1234.0f);
        fn(in.data(), actual.data(), rows, cols);
        for (std::size_t i = 0; i < in.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-6f)
                << name << " rows=" << rows << " cols=" << cols << " i=" << i;
        }
        expect_softmax_rows_normalized(actual, rows, cols);
    };
    if (has("avx2") && has("fma")) check(cpu::softmax_avx2, "softmax_avx2");
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        check(cpu::softmax_avx512, "softmax_avx512");
    }
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
    if (has("sse2")) {
        std::fill(relu_actual.begin(), relu_actual.end(), -1234.0f);
        cpu::relu_sse2(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }
    if (has("avx2") && has("fma")) {
        std::fill(relu_actual.begin(), relu_actual.end(), -1234.0f);
        cpu::relu_avx2(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        std::fill(relu_actual.begin(), relu_actual.end(), -1234.0f);
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

TEST(SimdKernelsTest, ReluVariantsCoverVectorLoopsAndTails) {
    // Widths around 4/8/16 lanes exercise SSE2, AVX2 and AVX-512 vector loops
    // plus their scalar tails. 2 * 33 is a row-major 2-D-shaped buffer.
    for (std::int64_t n : {1, 4, 7, 8, 15, 16, 17, 33, 65, 2 * 33}) {
        expect_relu_matches_scalar(n);
    }
}

TEST(SimdKernelsTest, SoftmaxTwoDimensionalRowsMatchScalarAndNormalize) {
    for (std::int64_t cols : {17, 33, 65}) {
        expect_softmax_matches_scalar(2, cols);
    }
}
