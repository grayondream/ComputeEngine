#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "backends/cpu/kernels/cpu_kernel_catalog.h"
#include "backends/cpu/simd/simd_kernels.h"
#include "core/hardware/detector.h"

namespace {

using RowFn = void (*)(const float*, const float*, float*, std::int64_t,
                       std::int64_t);

double seconds_per_call(RowFn fn, std::int64_t n, int reps) {
    const auto elements = static_cast<std::size_t>(n * n);
    std::vector<float> a(elements, 1.0f);
    std::vector<float> b(elements, 1.0f);
    std::vector<float> out(elements, 0.0f);

    auto run = [&] {
        for (std::int64_t row = 0; row < n; ++row) {
            fn(a.data() + row * n, b.data(), out.data() + row * n, n, n);
        }
    };

    run();  // warm up and fault in pages
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        run();
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    return elapsed / static_cast<double>(reps);
}

double gflops(std::int64_t n, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }
    const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                         static_cast<double>(n);
    return flops / seconds / 1e9;
}

}  // namespace

int main() {
    auto detected = ie::HardwareDetector::detect_cpu();
    if (!detected.has_value()) {
        std::cerr << "CPU detection failed: " << detected.error().message
                  << '\n';
        return 1;
    }
    const ie::HardwareProfile& profile = *detected;

    std::cout << "CPU profile: "
              << (profile.device_name.empty() ? "unknown" : profile.device_name)
              << '\n';
    std::cout << "vendor: "
              << (profile.vendor.empty() ? "unknown" : profile.vendor) << '\n';
    std::cout << "cores: " << profile.cu.num_cores
              << " threads_per_core: " << profile.cu.threads_per_core << '\n';

    auto selected = ie::CpuKernelCatalog::select(profile);
    if (!selected.has_value()) {
        std::cerr << "kernel selection failed: " << selected.error().message
                  << '\n';
        return 1;
    }
    std::cout << "selected variants: matmul=" << selected->matmul
              << " relu=" << selected->relu
              << " softmax=" << selected->softmax << '\n';

    constexpr int kReps = 5;
    std::cout << std::fixed << std::setprecision(2);
    for (std::int64_t n : {std::int64_t{64}, std::int64_t{128},
                           std::int64_t{256}}) {
        const double scalar_seconds =
            seconds_per_call(ie::cpu::matmul_row_scalar, n, kReps);
        const double selected_seconds =
            seconds_per_call(selected->fns.matmul_row, n, kReps);
        std::cout << "matmul " << n << "x" << n << "x" << n
                  << " scalar=" << gflops(n, scalar_seconds)
                  << " GFLOPS selected=" << gflops(n, selected_seconds)
                  << " GFLOPS";
        if (!selected->matmul.empty()) {
            std::cout << " (" << selected->matmul << ")";
        }
        std::cout << '\n';
    }
    return 0;
}
