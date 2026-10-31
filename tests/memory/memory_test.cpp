#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/memory/malloc_allocator.h"
#include "core/memory/tensor.h"
#include "core/memory/transfer_manager.h"

using namespace ie;

TEST(MallocAllocatorTest, AllocFreeTracksOutstanding) {
    MallocAllocator a;
    auto h = a.alloc(64, MemoryKind::HOST);
    ASSERT_TRUE(h.has_value());
    EXPECT_NE(*h, nullptr);
    EXPECT_EQ(a.outstanding_blocks(), 1u);
    a.free(*h);
    EXPECT_EQ(a.outstanding_blocks(), 0u);
}

TEST(MallocAllocatorTest, ZeroSizeIsInvalid) {
    MallocAllocator a;
    auto h = a.alloc(0, MemoryKind::HOST);
    ASSERT_FALSE(h.has_value());
    EXPECT_EQ(h.error().code, Error::Code::InvalidArgument);
}

TEST(MallocAllocatorTest, FreeDecrementsOutstandingBytes) {
    MallocAllocator a;
    auto h = a.alloc(64, MemoryKind::HOST);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(a.outstanding_bytes(), 64u);
    a.free(*h);
    EXPECT_EQ(a.outstanding_bytes(), 0u);
}

TEST(TensorTest, FromDataHasShapeAndValues) {
    Tensor t = Tensor::from_data({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(t.shape.numel(), 3);
    ASSERT_NE(t.data<float>(), nullptr);
    EXPECT_FLOAT_EQ(t.data<float>()[2], 3.0f);
}

TEST(TensorTest, CopyOutlivesOriginal) {
    Tensor copy;
    {
        Tensor original = Tensor::from_data({4.0f, 5.0f});
        copy = original;
    }
    EXPECT_FLOAT_EQ(copy.data<float>()[1], 5.0f);
}

TEST(TensorTest, AllocateReleasesThroughAllocator) {
    auto alloc = std::make_shared<MallocAllocator>();
    {
        auto tensor = Tensor::allocate(Shape{{4}}, DType::F32, alloc);
        ASSERT_TRUE(tensor.has_value());
        EXPECT_EQ(alloc->outstanding_blocks(), 1u);
    }
    EXPECT_EQ(alloc->outstanding_blocks(), 0u);
    EXPECT_EQ(alloc->outstanding_bytes(), 0u);
}

TEST(TensorTest, AllocateRejectsNonConcreteShapes) {
    auto alloc = std::make_shared<MallocAllocator>();
    const std::vector<Shape> shapes = {Shape{{-1}}, Shape{{-1, -1}},
                                       Shape{{0, 5}}, Shape{}};
    for (const Shape& shape : shapes) {
        auto tensor = Tensor::allocate(shape, DType::F32, alloc);
        ASSERT_FALSE(tensor.has_value()) << to_string(shape);
        EXPECT_EQ(tensor.error().code, Error::Code::InvalidArgument);
    }
}

TEST(TransferManagerTest, SameDeviceCopySucceeds) {
    TransferManager tm;
    EXPECT_TRUE(tm.copy(Device::CPU, Device::CPU, nullptr, 0).has_value());
    EXPECT_EQ(tm.copy(Device::CPU, Device::CUDA, nullptr, 0).error().code,
              Error::Code::Unsupported);
}
