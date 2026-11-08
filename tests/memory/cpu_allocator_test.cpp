#include <gtest/gtest.h>

#include <cstdint>

#include "backends/cpu/cpu_allocator.h"

using namespace ie;

TEST(CpuAllocatorTest, AlignsAndTracksBytes) {
    CpuAllocator alloc;
    auto ptr = alloc.alloc(65, MemoryKind::HOST);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % CpuAllocator::kAlignment,
              0u);
    EXPECT_EQ(alloc.outstanding_blocks(), 1u);
    EXPECT_GE(alloc.outstanding_bytes(), 65u);
    alloc.free(*ptr);
    EXPECT_EQ(alloc.outstanding_blocks(), 0u);
    EXPECT_EQ(alloc.outstanding_bytes(), 0u);
    alloc.free(*ptr);
    alloc.free(nullptr);
}

TEST(CpuAllocatorTest, RejectsZeroBytes) {
    CpuAllocator alloc;
    auto ptr = alloc.alloc(0, MemoryKind::HOST);
    ASSERT_FALSE(ptr.has_value());
    EXPECT_EQ(ptr.error().code, Error::Code::InvalidArgument);
}
