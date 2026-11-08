#pragma once

#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/hardware/profile.h"
#include "core/ir/op_type.h"
#include "core/tuning/kernel_variant.h"
#include "core/tuning/kernel_variant_registry.h"

namespace ie {

// Registers the F32 CPU variants for MatMul/Relu/Softmax. Scalar has no
// capability requirements; each higher tier requires the ISA features its
// kernels are compiled for. Cost priorities are 1/2/4/8 estimated TFLOPS.
inline void register_cpu_kernel_variants(KernelVariantRegistry& registry) {
    const struct Tier {
        const char* suffix;
        std::vector<std::string> caps;
        float tflops;
    } tiers[] = {
        {"scalar", {}, 1.0f},
        {"sse2", {"sse2"}, 2.0f},
        {"avx2", {"avx2", "fma"}, 4.0f},
        {"avx512", {"avx512f", "avx512bw", "avx512vl"}, 8.0f},
    };

    const OpType ops[] = {OpType::MatMul, OpType::Relu, OpType::Softmax};
    const char* prefixes[] = {"matmul_", "relu_", "softmax_"};

    for (std::size_t i = 0; i < std::size(ops); ++i) {
        for (const Tier& tier : tiers) {
            KernelVariant variant;
            variant.name = std::string(prefixes[i]) + tier.suffix;
            variant.target_kind = ModelKind::UNKNOWN;
            variant.req.required_caps = tier.caps;
            variant.cost.estimated_tflops = tier.tflops;
            registry.register_variant(ops[i], DType::F32, std::move(variant));
        }
    }
}

namespace detail {

// Shared, initialized-once registry for the core CPU variants.
inline const KernelVariantRegistry& cpu_kernel_variant_registry() {
    static const KernelVariantRegistry registry = [] {
        KernelVariantRegistry built;
        register_cpu_kernel_variants(built);
        return built;
    }();
    return registry;
}

}  // namespace detail

// Picks the matching variant with the highest estimated TFLOPS. Scalar is
// always registered, so a valid (op, dtype) combination always yields a match.
inline Result<KernelVariant> select_best_cpu_variant(
    const KernelVariantRegistry& registry, OpType op, DType dtype,
    const Shape& shape, const HardwareProfile& profile) {
    Result<std::vector<KernelVariant>> matches =
        registry.select(op, dtype, shape, profile);
    if (!matches.has_value()) {
        return matches.error();
    }

    const KernelVariant* best = nullptr;
    for (const KernelVariant& variant : *matches) {
        if (best == nullptr ||
            variant.cost.estimated_tflops > best->cost.estimated_tflops) {
            best = &variant;
        }
    }
    if (best == nullptr) {
        return Error::not_found("no CPU kernel variant matches");
    }
    return *best;
}

inline Result<KernelVariant> select_best_cpu_variant(
    OpType op, DType dtype, const Shape& shape,
    const HardwareProfile& profile) {
    return select_best_cpu_variant(detail::cpu_kernel_variant_registry(), op,
                                   dtype, shape, profile);
}

}  // namespace ie
