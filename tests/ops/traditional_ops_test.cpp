#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/memory/malloc_allocator.h"
#include "core/ops/op_registry.h"
#include "data/traditional_reference.h"
#include "ops/traditional/batch_norm_op.h"
#include "ops/traditional/conv2d_op.h"
#include "ops/traditional/pool_ops.h"

using namespace ie;

namespace {

Tensor make_tensor(const Shape& shape, const float* data) {
    auto allocator = std::make_shared<MallocAllocator>();
    auto tensor = Tensor::allocate(shape, DType::F32, allocator);
    if (!tensor.has_value()) {
        return {};
    }
    for (int64_t i = 0; i < shape.numel(); ++i) {
        tensor->data<float>()[i] = data[static_cast<std::size_t>(i)];
    }
    return std::move(*tensor);
}

ExecutionContext make_context() {
    ExecutionContext context;
    context.allocator = std::make_shared<MallocAllocator>();
    return context;
}

std::atomic<int> g_row_calls{0};

void counting_matmul_row(const float* a_row, const float* b, float* out_row,
                         std::int64_t k, std::int64_t n) {
    ++g_row_calls;
    for (std::int64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (std::int64_t t = 0; t < k; ++t) {
            acc += a_row[t] * b[t * n + j];
        }
        out_row[j] = acc;
    }
}

// Deterministic input used by the padding and ceil-mode pool tests.
Tensor ramp_tensor(int64_t h, int64_t w) {
    std::vector<float> values(static_cast<std::size_t>(h * w));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    return make_tensor(Shape{{1, 1, h, w}}, values.data());
}

}  // namespace

TEST(TraditionalOpsTest, Conv2DMatchesReference) {
    Conv2DOp op;
    Attrs attrs;
    attrs["strides"] = std::vector<int64_t>{1, 1};
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    auto w = make_tensor(Shape{{3, 2, 3, 3}}, kConvWeight);
    auto b = make_tensor(Shape{{3}}, kConvBias);
    std::vector<Tensor*> inputs{&x, &w, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 3, 5, 5}}));
    constexpr std::size_t kSize = 75;
    for (std::size_t i = 0; i < kSize; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kConvExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, Conv2DWithoutBiasMatchesReference) {
    Conv2DOp op;
    Attrs attrs;
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    auto w = make_tensor(Shape{{3, 2, 3, 3}}, kConvWeight);
    std::vector<Tensor*> inputs{&x, &w};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 3, 5, 5}}));
    for (int64_t m = 0; m < 3; ++m) {
        for (int64_t p = 0; p < 25; ++p) {
            const std::size_t index = static_cast<std::size_t>(m * 25 + p);
            EXPECT_NEAR(out[0].data<float>()[index],
                        kConvExpected[index] - kConvBias[m], 1e-5f);
        }
    }
}

TEST(TraditionalOpsTest, Conv2DUsesInjectedRowKernel) {
    Conv2DOp op;
    Attrs attrs;
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    auto w = make_tensor(Shape{{3, 2, 3, 3}}, kConvWeight);
    auto b = make_tensor(Shape{{3}}, kConvBias);
    KernelFns fns;
    fns.matmul_row = counting_matmul_row;
    ExecutionContext ctx = make_context();
    ctx.kernels = &fns;
    g_row_calls = 0;

    std::vector<Tensor*> inputs{&x, &w, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    // One row call per output channel for a single batch item.
    EXPECT_EQ(g_row_calls.load(), 3);
    for (std::size_t i = 0; i < 75; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kConvExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, Conv2DSupportsSameUpperAutoPad) {
    Conv2DOp op;
    Attrs attrs;
    attrs["auto_pad"] = std::string("SAME_UPPER");
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    auto w = make_tensor(Shape{{3, 2, 3, 3}}, kConvWeight);
    auto b = make_tensor(Shape{{3}}, kConvBias);
    std::vector<Tensor*> inputs{&x, &w, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 3, 5, 5}}));
    for (std::size_t i = 0; i < 75; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kConvExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, Conv2DGroupedMatchesManualValues) {
    Conv2DOp op;
    Attrs attrs;
    attrs["group"] = static_cast<int64_t>(2);
    ASSERT_TRUE(op.configure(attrs).has_value());

    const float x_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float w_data[2] = {2, 3};
    auto x = make_tensor(Shape{{1, 2, 2, 2}}, x_data);
    auto w = make_tensor(Shape{{2, 1, 1, 1}}, w_data);
    std::vector<Tensor*> inputs{&x, &w};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 2, 2, 2}}));
    const float expected[8] = {2, 4, 6, 8, 15, 18, 21, 24};
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out[0].data<float>()[i], expected[i]);
    }
}

TEST(TraditionalOpsTest, Conv2DRejectsInvalidInputs) {
    Conv2DOp op;
    Attrs attrs;
    attrs["group"] = static_cast<int64_t>(4);
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    auto w = make_tensor(Shape{{3, 2, 3, 3}}, kConvWeight);
    ExecutionContext ctx = make_context();
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    std::vector<Tensor*> inputs{&x, &w};
    EXPECT_EQ(op.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    Conv2DOp default_op;
    ASSERT_TRUE(default_op.configure(Attrs{}).has_value());
    auto half = make_tensor(Shape{{1, 2, 5, 5}}, kConvInput);
    half.dtype = DType::F16;
    std::vector<Tensor*> half_inputs{&half, &w};
    EXPECT_EQ(default_op.forward(half_inputs, outputs, ctx).error().code,
              Error::Code::Unsupported);

    std::vector<Tensor*> no_outputs;
    std::vector<Tensor*> ok_inputs{&x, &w};
    EXPECT_EQ(default_op.forward(ok_inputs, no_outputs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(TraditionalOpsTest, DepthwiseConv2DMatchesReference) {
    DepthwiseConv2DOp op;
    Attrs attrs;
    attrs["group"] = static_cast<int64_t>(2);
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 3, 3}}, kDepthwiseInput);
    auto w = make_tensor(Shape{{2, 1, 2, 2}}, kDepthwiseWeight);
    auto b = make_tensor(Shape{{2}}, kDepthwiseBias);
    std::vector<Tensor*> inputs{&x, &w, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 2, 2, 2}}));
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kDepthwiseExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, DepthwiseConv2DRequiresChannelGroup) {
    DepthwiseConv2DOp op;
    Attrs attrs;
    attrs["group"] = static_cast<int64_t>(1);
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 2, 3, 3}}, kDepthwiseInput);
    auto w = make_tensor(Shape{{2, 1, 2, 2}}, kDepthwiseWeight);
    std::vector<Tensor*> inputs{&x, &w};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    EXPECT_EQ(op.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(TraditionalOpsTest, BatchNormMatchesReference) {
    BatchNormOp op;
    Attrs attrs;
    attrs["epsilon"] = 1e-5;
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor(Shape{{1, 3, 2, 2}}, kBatchNormInput);
    auto scale = make_tensor(Shape{{3}}, kBatchNormScale);
    auto bias = make_tensor(Shape{{3}}, kBatchNormBias);
    auto mean = make_tensor(Shape{{3}}, kBatchNormMean);
    auto var = make_tensor(Shape{{3}}, kBatchNormVar);
    std::vector<Tensor*> inputs{&x, &scale, &bias, &mean, &var};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 3, 2, 2}}));
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kBatchNormExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, BatchNormRejectsWrongInputCount) {
    BatchNormOp op;
    ASSERT_TRUE(op.configure(Attrs{}).has_value());
    auto x = make_tensor(Shape{{1, 3, 2, 2}}, kBatchNormInput);
    std::vector<Tensor*> inputs{&x};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    EXPECT_EQ(op.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(TraditionalOpsTest, Conv2DAutoPadDirectionAndDilation) {
    // 4x4 ramp, kernel 3, stride 2: SAME modes need one asymmetric pad.
    std::vector<float> values(16);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    auto x = make_tensor(Shape{{1, 1, 4, 4}}, values.data());
    const float weight_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto w = make_tensor(Shape{{1, 1, 3, 3}}, weight_data);
    ExecutionContext ctx = make_context();

    const auto run = [&](const char* auto_pad) {
        Conv2DOp op;
        Attrs attrs;
        attrs["auto_pad"] = std::string(auto_pad);
        attrs["strides"] = std::vector<int64_t>{2, 2};
        EXPECT_TRUE(op.configure(attrs).has_value());
        std::vector<Tensor*> inputs{&x, &w};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_TRUE(op.forward(inputs, outputs, ctx).has_value());
        return out[0];
    };

    Tensor upper = run("SAME_UPPER");  // pads {0,0,1,1}
    Tensor lower = run("SAME_LOWER");  // pads {1,1,0,0}
    ASSERT_EQ(upper.shape, (Shape{{1, 1, 2, 2}}));
    ASSERT_EQ(lower.shape, (Shape{{1, 1, 2, 2}}));
    EXPECT_FLOAT_EQ(upper.data<float>()[0], 1 + 2 + 3 + 5 + 6 + 7 + 9 + 10 + 11);
    EXPECT_FLOAT_EQ(lower.data<float>()[3], 6 + 7 + 8 + 10 + 11 + 12 + 14 + 15 + 16);

    // Dilation 2 on a 3x3 input: effective 3x3 window covers the corners.
    Conv2DOp dilated;
    Attrs dilation_attrs;
    dilation_attrs["dilations"] = std::vector<int64_t>{2, 2};
    ASSERT_TRUE(dilated.configure(dilation_attrs).has_value());
    auto small_values = ramp_tensor(3, 3);
    auto unit = make_tensor(Shape{{1, 1, 2, 2}},
                            std::vector<float>(4, 1.0f).data());
    std::vector<Tensor*> inputs{&small_values, &unit};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ASSERT_TRUE(dilated.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 1, 1, 1}}));
    EXPECT_FLOAT_EQ(out[0].data<float>()[0], 1 + 3 + 7 + 9);
}

TEST(TraditionalOpsTest, PoolAndGlobalAvgMatchReference) {
    auto x = make_tensor(Shape{{1, 2, 4, 4}}, kPoolInput);

    MaxPool2DOp max_op;
    Attrs attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{2, 2};
    attrs["strides"] = std::vector<int64_t>{2, 2};
    ASSERT_TRUE(max_op.configure(attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ExecutionContext ctx = make_context();
        ASSERT_TRUE(max_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 2, 2, 2}}));
        for (std::size_t i = 0; i < 8; ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], kMaxPoolExpected[i]);
        }
    }

    AvgPool2DOp avg_op;
    ASSERT_TRUE(avg_op.configure(attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ExecutionContext ctx = make_context();
        ASSERT_TRUE(avg_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 2, 2, 2}}));
        for (std::size_t i = 0; i < 8; ++i) {
            EXPECT_NEAR(out[0].data<float>()[i], kAvgPoolExpected[i], 1e-6f);
        }
    }

    GlobalAvgPoolOp global_op;
    auto gx = make_tensor(Shape{{1, 2, 3, 2}}, kGlobalAvgInput);
    {
        std::vector<Tensor*> inputs{&gx};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ExecutionContext ctx = make_context();
        ASSERT_TRUE(global_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 2, 1, 1}}));
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(out[0].data<float>()[i], kGlobalAvgExpected[i], 1e-6f);
        }
    }
}

TEST(TraditionalOpsTest, PoolPaddingAndCountIncludePad) {
    auto x = ramp_tensor(3, 3);
    ExecutionContext ctx = make_context();

    MaxPool2DOp max_op;
    Attrs max_attrs;
    max_attrs["kernel_shape"] = std::vector<int64_t>{2, 2};
    max_attrs["strides"] = std::vector<int64_t>{2, 2};
    max_attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(max_op.configure(max_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(max_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1, 2, 2}}));
        const float expected[4] = {1.0f, 3.0f, 7.0f, 9.0f};
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], expected[i]);
        }
    }

    AvgPool2DOp avg_op;
    ASSERT_TRUE(avg_op.configure(max_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(avg_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1, 2, 2}}));
        const float expected[4] = {1.0f, 2.5f, 5.5f, 7.0f};
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(out[0].data<float>()[i], expected[i], 1e-6f);
        }
    }

    Attrs include_pad = max_attrs;
    include_pad["count_include_pad"] = static_cast<int64_t>(1);
    ASSERT_TRUE(avg_op.configure(include_pad).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(avg_op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1, 2, 2}}));
        const float expected[4] = {0.25f, 1.25f, 2.75f, 7.0f};
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(out[0].data<float>()[i], expected[i], 1e-6f);
        }
    }
}

TEST(TraditionalOpsTest, PoolCeilModeAddsTrailingWindow) {
    auto x = ramp_tensor(5, 5);
    ExecutionContext ctx = make_context();

    MaxPool2DOp max_op;
    Attrs attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{2, 2};
    attrs["strides"] = std::vector<int64_t>{2, 2};
    attrs["ceil_mode"] = static_cast<int64_t>(1);
    ASSERT_TRUE(max_op.configure(attrs).has_value());
    std::vector<Tensor*> inputs{&x};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ASSERT_TRUE(max_op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 1, 3, 3}}));
    EXPECT_FLOAT_EQ(out[0].data<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(out[0].data<float>()[8], 25.0f);

    AvgPool2DOp avg_op;
    ASSERT_TRUE(avg_op.configure(attrs).has_value());
    std::vector<Tensor> avg_out(1);
    std::vector<Tensor*> avg_outputs{&avg_out[0]};
    ASSERT_TRUE(avg_op.forward(inputs, avg_outputs, ctx).has_value());
    ASSERT_EQ(avg_out[0].shape, (Shape{{1, 1, 3, 3}}));
    EXPECT_NEAR(avg_out[0].data<float>()[8], 25.0f, 1e-6f);
}

TEST(TraditionalOpsTest, PoolRequiresKernelShapeAndF32) {
    MaxPool2DOp op;
    EXPECT_EQ(op.configure(Attrs{}).error().code,
              Error::Code::InvalidArgument);

    AvgPool2DOp avg_op;
    Attrs attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{2, 2};
    ASSERT_TRUE(avg_op.configure(attrs).has_value());
    auto x = make_tensor(Shape{{1, 1, 2, 2}}, kPoolInput);
    x.dtype = DType::F16;
    std::vector<Tensor*> inputs{&x};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    EXPECT_EQ(avg_op.forward(inputs, outputs, ctx).error().code,
              Error::Code::Unsupported);
}

TEST(TraditionalOpsTest, RegistryCreatesTraditionalOps) {
    auto& registry = OpRegistry::instance();
    register_builtin_ops(registry);

    for (OpType type : {OpType::Conv2D, OpType::DepthwiseConv2D,
                        OpType::MaxPool2D, OpType::AvgPool2D,
                        OpType::GlobalAvgPool, OpType::BatchNorm}) {
        auto op = registry.create(type);
        ASSERT_TRUE(op.has_value());
        EXPECT_NE(op->get(), nullptr);
    }
    auto conv = registry.create(OpType::Conv2D);
    ASSERT_TRUE(conv.has_value());
    EXPECT_NE(dynamic_cast<Conv2DOp*>(conv->get()), nullptr);
    auto pool = registry.create(OpType::MaxPool2D);
    ASSERT_TRUE(pool.has_value());
    EXPECT_NE(dynamic_cast<MaxPool2DOp*>(pool->get()), nullptr);
    auto norm = registry.create(OpType::BatchNorm);
    ASSERT_TRUE(norm.has_value());
    EXPECT_NE(dynamic_cast<BatchNormOp*>(norm->get()), nullptr);
}
