#include "backends/cpu/kernels/cpu_kernel_catalog.h"

#include <string>

#include "backends/cpu/simd/simd_kernels.h"
#include "core/base/shape.h"
#include "core/tuning/cpu_kernel_variants.h"
#include "core/tuning/kernel_variant_registry.h"

namespace ie {
namespace {

const KernelVariantRegistry& kernel_registry() {
    static const KernelVariantRegistry registry = [] {
        KernelVariantRegistry built;
        register_cpu_kernel_variants(built);
        return built;
    }();
    return registry;
}

KernelFns::MatMulRowFn matmul_row_for(const std::string& name) {
    if (name == "matmul_scalar") return cpu::matmul_row_scalar;
    if (name == "matmul_sse2") return cpu::matmul_row_sse2;
    if (name == "matmul_avx2") return cpu::matmul_row_avx2;
    if (name == "matmul_avx512") return cpu::matmul_row_avx512;
    return nullptr;
}

KernelFns::UnaryFn relu_for(const std::string& name) {
    if (name == "relu_scalar") return cpu::relu_scalar;
    if (name == "relu_sse2") return cpu::relu_sse2;
    if (name == "relu_avx2") return cpu::relu_avx2;
    if (name == "relu_avx512") return cpu::relu_avx512;
    return nullptr;
}

KernelFns::SoftmaxFn softmax_for(const std::string& name) {
    if (name == "softmax_scalar") return cpu::softmax_scalar;
    // There is no dedicated SSE2 softmax kernel; exp dominates this op, so the
    // SSE2 tier intentionally reuses the scalar implementation.
    if (name == "softmax_sse2") return cpu::softmax_scalar;
    if (name == "softmax_avx2") return cpu::softmax_avx2;
    if (name == "softmax_avx512") return cpu::softmax_avx512;
    return nullptr;
}

}  // namespace

Result<CpuKernelCatalog::Selection> CpuKernelCatalog::select(
    const HardwareProfile& profile) {
    const Shape shape{};
    const KernelVariantRegistry& registry = kernel_registry();

    Result<KernelVariant> matmul = select_best_cpu_variant(
        registry, OpType::MatMul, DType::F32, shape, profile);
    if (!matmul.has_value()) {
        return matmul.error();
    }
    Result<KernelVariant> relu = select_best_cpu_variant(
        registry, OpType::Relu, DType::F32, shape, profile);
    if (!relu.has_value()) {
        return relu.error();
    }
    Result<KernelVariant> softmax = select_best_cpu_variant(
        registry, OpType::Softmax, DType::F32, shape, profile);
    if (!softmax.has_value()) {
        return softmax.error();
    }

    Selection selection;
    selection.matmul = matmul->name;
    selection.relu = relu->name;
    selection.softmax = softmax->name;
    selection.fns.matmul_row = matmul_row_for(selection.matmul);
    selection.fns.relu = relu_for(selection.relu);
    selection.fns.softmax = softmax_for(selection.softmax);

    if (selection.fns.matmul_row == nullptr || selection.fns.relu == nullptr ||
        selection.fns.softmax == nullptr) {
        return Error::not_found("CPU kernel variant has no implementation");
    }
    return selection;
}

}  // namespace ie
