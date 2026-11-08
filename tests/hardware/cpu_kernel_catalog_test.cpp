#include <gtest/gtest.h>

#include <initializer_list>

#include "backends/cpu/kernels/cpu_kernel_catalog.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/hardware/profile.h"
#include "core/ir/op_type.h"
#include "core/tuning/cpu_kernel_variants.h"

using namespace ie;

namespace {
HardwareProfile profile_with(std::initializer_list<const char*> caps) {
    HardwareProfile p;
    for (const char* cap : caps) {
        p.caps.set(cap, true);
    }
    return p;
}
}  // namespace

TEST(CpuKernelCatalogTest, SelectsBestVariantByCaps) {
    Shape shape{{16, 16}};
    auto scalar = select_best_cpu_variant(OpType::MatMul, DType::F32, shape,
                                          HardwareProfile{});
    ASSERT_TRUE(scalar.has_value());
    EXPECT_EQ(scalar->name, "matmul_scalar");

    auto sse2 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"sse2"}));
    ASSERT_TRUE(sse2.has_value());
    EXPECT_EQ(sse2->name, "matmul_sse2");

    auto avx2 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"avx2", "fma"}));
    ASSERT_TRUE(avx2.has_value());
    EXPECT_EQ(avx2->name, "matmul_avx2");

    auto avx512 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape,
        profile_with({"avx2", "fma", "avx512f", "avx512bw", "avx512vl"}));
    ASSERT_TRUE(avx512.has_value());
    EXPECT_EQ(avx512->name, "matmul_avx512");
}

TEST(CpuKernelCatalogTest, Avx2RequiresFmaAndAvx512RequiresBwVl) {
    Shape shape{{16, 16}};
    auto no_fma = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"sse2", "avx2"}));
    ASSERT_TRUE(no_fma.has_value());
    EXPECT_EQ(no_fma->name, "matmul_sse2");

    auto partial_512 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"avx2", "fma", "avx512f"}));
    ASSERT_TRUE(partial_512.has_value());
    EXPECT_EQ(partial_512->name, "matmul_avx2");
}

TEST(CpuKernelCatalogTest, SelectionFillsFunctionPointers) {
    auto selection = CpuKernelCatalog::select(profile_with({"sse2"}));
    ASSERT_TRUE(selection.has_value());
    EXPECT_EQ(selection->matmul, "matmul_sse2");
    EXPECT_NE(selection->fns.matmul_row, nullptr);
    EXPECT_NE(selection->fns.relu, nullptr);
    EXPECT_NE(selection->fns.softmax, nullptr);
}
