#include <gtest/gtest.h>

#include "core/memory/malloc_allocator.h"
#include "core/ops/common/add_op.h"
#include "core/ops/common/matmul_op.h"
#include "core/ops/common/mul_op.h"
#include "core/ops/op_registry.h"

using namespace ie;

TEST(OpRegistryTest, RegisterCreateAndLookup) {
    auto& reg = OpRegistry::instance();
    register_builtin_ops(reg);
    auto op = reg.create(OpType::Add);
    ASSERT_TRUE(op.has_value());
    EXPECT_NE(dynamic_cast<AddOp*>(op->get()), nullptr);
    EXPECT_FALSE(reg.list_by_kind(ModelKind::TRADITIONAL).empty());
}

TEST(OpRegistryTest, UnknownOpReturnsNotFound) {
    auto& reg = OpRegistry::instance();
    auto op = reg.create(OpType::Conv2D);
    ASSERT_FALSE(op.has_value());
    EXPECT_EQ(op.error().code, Error::Code::NotFound);
}

TEST(OpRegistryTest, DuplicateRegistrationIsRejected) {
    auto& reg = OpRegistry::instance();
    auto first = reg.register_op(OpType::Sub,
                                 [] { return std::make_unique<AddOp>(); });
    EXPECT_TRUE(first.has_value() ||
                first.error().code == Error::Code::AlreadyExists);
    auto second = reg.register_op(OpType::Sub,
                                  [] { return std::make_unique<AddOp>(); });
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, Error::Code::AlreadyExists);
}

TEST(AddOpTest, AddsElementwiseAndRejectsShapeMismatch) {
    MallocAllocator alloc;
    ExecutionContext ctx;
    ctx.allocator = &alloc;
    AddOp op;
    Tensor a = Tensor::from_data({1, 2, 3});
    Tensor b = Tensor::from_data({4, 5, 6});
    Tensor out;
    std::vector<Tensor*> in{&a, &b};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 5.0f);

    Tensor c = Tensor::from_data({7, 8});
    std::vector<Tensor*> bad{&a, &c};
    EXPECT_EQ(op.forward(bad, outs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(MulOpTest, ScalarMultiply) {
    MallocAllocator alloc;
    ExecutionContext ctx;
    ctx.allocator = &alloc;
    MulOp op;
    Attrs attrs{{"scalar", 3.0}};
    ASSERT_TRUE(op.configure(attrs).has_value());
    Tensor a = Tensor::from_data({1, 2, 3});
    Tensor out;
    std::vector<Tensor*> in{&a};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[2], 9.0f);
}

TEST(MatMulOpTest, TwoDimensionalProduct) {
    MallocAllocator alloc;
    ExecutionContext ctx;
    ctx.allocator = &alloc;
    MatMulOp op;
    Tensor a = Tensor::from_data({1, 2, 3, 4});
    a.shape = Shape{{2, 2}};
    Tensor b = Tensor::from_data({5, 6, 7, 8});
    b.shape = Shape{{2, 2}};
    Tensor out;
    std::vector<Tensor*> in{&a, &b};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    ASSERT_EQ(out.shape.dims, (std::vector<int64_t>{2, 2}));
    EXPECT_FLOAT_EQ(out.data<float>()[0], 19.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 50.0f);
}
