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
#include "ops/traditional/activation_ops.h"
#include "ops/traditional/batch_norm_op.h"
#include "ops/traditional/conv2d_op.h"
#include "ops/traditional/data_ops.h"
#include "ops/traditional/gemm_op.h"
#include "ops/traditional/layout_ops.h"
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

// Compares a producing tensor against exact or near expected values.
void expect_values(const Tensor& tensor, const std::vector<float>& expected,
                   float tolerance = 0.0f) {
    ASSERT_EQ(tensor.shape.numel(), static_cast<int64_t>(expected.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (tolerance == 0.0f) {
            EXPECT_FLOAT_EQ(tensor.data<float>()[i], expected[i]);
        } else {
            EXPECT_NEAR(tensor.data<float>()[i], expected[i], tolerance);
        }
    }
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

TEST(TraditionalOpsTest, PoolSameUpperMatchesExplicitPads) {
    auto x = ramp_tensor(3, 3);
    ExecutionContext ctx = make_context();

    const auto run = [&](const Attrs& attrs) {
        MaxPool2DOp op;
        EXPECT_TRUE(op.configure(attrs).has_value());
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_TRUE(op.forward(inputs, outputs, ctx).has_value());
        return out[0];
    };

    Attrs same;
    same["kernel_shape"] = std::vector<int64_t>{2, 2};
    same["strides"] = std::vector<int64_t>{2, 2};
    same["auto_pad"] = std::string("SAME_UPPER");
    Attrs explicit_upper = same;
    explicit_upper.erase("auto_pad");
    explicit_upper["pads"] = std::vector<int64_t>{0, 0, 1, 1};
    Attrs same_lower = same;
    same_lower["auto_pad"] = std::string("SAME_LOWER");
    Attrs explicit_lower = explicit_upper;
    explicit_lower["pads"] = std::vector<int64_t>{1, 1, 0, 0};

    Tensor upper = run(same);
    Tensor upper_ref = run(explicit_upper);
    ASSERT_EQ(upper.shape, upper_ref.shape);
    ASSERT_EQ(upper.shape, (Shape{{1, 1, 2, 2}}));
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(upper.data<float>()[i], upper_ref.data<float>()[i]);
    }

    Tensor lower = run(same_lower);
    Tensor lower_ref = run(explicit_lower);
    ASSERT_EQ(lower.shape, lower_ref.shape);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(lower.data<float>()[i], lower_ref.data<float>()[i]);
    }
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

TEST(TraditionalOpsTest, GemmMatchesReferenceWithTransposeAndBias) {
    GemmOp op;
    Attrs attrs;
    attrs["transB"] = static_cast<int64_t>(1);
    ASSERT_TRUE(op.configure(attrs).has_value());

    const float a_data[6] = {1, 2, 3, 4, 5, 6};
    const float b_data[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2};
    const float c_data[4] = {0.5f, -1.0f, 2.0f, 3.0f};
    auto a = make_tensor(Shape{{2, 3}}, a_data);
    auto b = make_tensor(Shape{{4, 3}}, b_data);
    auto c = make_tensor(Shape{{4}}, c_data);
    std::vector<Tensor*> inputs{&a, &b, &c};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{2, 4}}));
    // Y[i][j] = sum_k A[i][k] * B[j][k] + C[j].
    const float expected[8] = {1.5f, 1.0f, 5.0f, 15.0f,
                               4.5f, 4.0f, 8.0f, 33.0f};
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], expected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, GemmWithoutBiasScalesByAlpha) {
    GemmOp op;
    Attrs attrs;
    attrs["alpha"] = 2.0;
    ASSERT_TRUE(op.configure(attrs).has_value());

    const float a_data[4] = {1, 2, 3, 4};
    const float b_data[4] = {1, 0, 1, 2};
    auto a = make_tensor(Shape{{2, 2}}, a_data);
    auto b = make_tensor(Shape{{2, 2}}, b_data);
    // No C input: the optional bias term must be omitted entirely.
    std::vector<Tensor*> inputs{&a, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx = make_context();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{2, 2}}));
    // Y = 2 * A * B, with no bias added.
    const float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[0].data<float>()[i], expected[i]);
    }
}

TEST(TraditionalOpsTest, GemmSupportsTransposeAAndBiasBroadcast) {
    // Physical A [3,2] with transA=1 is logically [2,3]; B is [3,3], so M=2 and
    // N=3 keep the [M] and [N] bias forms unambiguous.
    const float a_data[6] = {1, 4, 2, 5, 3, 6};
    const float b_data[9] = {1, 0, 0, 0, 1, 0, 1, 1, 1};
    auto a = make_tensor(Shape{{3, 2}}, a_data);
    auto b = make_tensor(Shape{{3, 3}}, b_data);
    ExecutionContext ctx = make_context();

    const auto run = [&](std::vector<int64_t> c_shape,
                         const std::vector<float>& c_values, double alpha,
                         double beta) {
        GemmOp op;
        Attrs attrs;
        attrs["transA"] = static_cast<int64_t>(1);
        attrs["alpha"] = alpha;
        attrs["beta"] = beta;
        EXPECT_TRUE(op.configure(attrs).has_value());
        auto c = make_tensor(Shape{std::move(c_shape)}, c_values.data());
        std::vector<Tensor*> inputs{&a, &b, &c};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_TRUE(op.forward(inputs, outputs, ctx).has_value());
        return out[0];
    };

    // C broadcast per row [M], alpha=2, beta=0.5.
    Tensor row = run({2}, {2.0f, 3.0f}, 2.0, 0.5);
    ASSERT_EQ(row.shape, (Shape{{2, 3}}));
    const float row_expected[6] = {9.0f, 11.0f, 7.0f,
                                   21.5f, 23.5f, 13.5f};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(row.data<float>()[i], row_expected[i], 1e-5f);
    }

    // C broadcast per column [N].
    Tensor column = run({3}, {1.0f, 2.0f, 3.0f}, 1.0, 1.0);
    ASSERT_EQ(column.shape, (Shape{{2, 3}}));
    const float column_expected[6] = {5.0f, 7.0f, 6.0f,
                                      11.0f, 13.0f, 9.0f};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(column.data<float>()[i], column_expected[i], 1e-5f);
    }

    // Scalar C broadcast.
    Tensor scalar = run({1}, {5.0f}, 1.0, 1.0);
    ASSERT_EQ(scalar.shape, (Shape{{2, 3}}));
    const float scalar_expected[6] = {9.0f, 10.0f, 8.0f,
                                      15.0f, 16.0f, 11.0f};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(scalar.data<float>()[i], scalar_expected[i], 1e-5f);
    }

    // Full [M,N] bias.
    Tensor full = run({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 1.0,
                      1.0);
    ASSERT_EQ(full.shape, (Shape{{2, 3}}));
    const float full_expected[6] = {5.0f, 7.0f, 6.0f,
                                    14.0f, 16.0f, 12.0f};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(full.data<float>()[i], full_expected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, GemmRejectsInvalidInputs) {
    GemmOp op;
    Attrs attrs;
    attrs["transB"] = static_cast<int64_t>(1);
    ASSERT_TRUE(op.configure(attrs).has_value());

    const float a_data[6] = {1, 2, 3, 4, 5, 6};
    const float b_data[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2};
    auto a = make_tensor(Shape{{2, 3}}, a_data);
    auto b = make_tensor(Shape{{4, 3}}, b_data);
    ExecutionContext ctx = make_context();
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};

    auto b_bad = make_tensor(Shape{{4, 2}}, b_data);
    std::vector<Tensor*> mismatch{&a, &b_bad};
    EXPECT_EQ(op.forward(mismatch, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    const float c_bad_data[3] = {1.0f, 2.0f, 3.0f};
    auto c_bad = make_tensor(Shape{{3}}, c_bad_data);
    std::vector<Tensor*> bad_bias{&a, &b, &c_bad};
    EXPECT_EQ(op.forward(bad_bias, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    auto half = a;
    half.dtype = DType::F16;
    std::vector<Tensor*> half_inputs{&half, &b};
    EXPECT_EQ(op.forward(half_inputs, outputs, ctx).error().code,
              Error::Code::Unsupported);

    std::vector<Tensor*> too_few{&a};
    EXPECT_EQ(op.forward(too_few, outputs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(TraditionalOpsTest, FlattenReshapeTransposePreserveValues) {
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    auto x = make_tensor(Shape{{2, 3, 2}}, values.data());
    ExecutionContext ctx = make_context();

    // Flatten axis=1 on [2,3,2] -> [2,6].
    FlattenOp flatten;
    ASSERT_TRUE(flatten.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(flatten.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{2, 6}}));
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], values[i]);
        }
    }

    // Flatten with a negative axis: -1 on [2,3,2] -> [6,2].
    FlattenOp flatten_negative;
    Attrs negative_axis;
    negative_axis["axis"] = static_cast<int64_t>(-1);
    ASSERT_TRUE(flatten_negative.configure(negative_axis).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(flatten_negative.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{6, 2}}));
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], values[i]);
        }
    }

    // Reshape [2,3,2] with shape {3,-1} -> [3,4].
    ReshapeOp reshape;
    Attrs shape_attrs;
    shape_attrs["shape"] = std::vector<int64_t>{3, -1};
    ASSERT_TRUE(reshape.configure(shape_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(reshape.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 4}}));
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], values[i]);
        }
    }

    // Reshape from an input shape tensor with a 0 copy-dim: {0,-1} -> [2,6].
    ReshapeOp reshape_input;
    ASSERT_TRUE(reshape_input.configure(Attrs{}).has_value());
    const float shape_data[2] = {0.0f, -1.0f};
    auto shape_tensor = make_tensor(Shape{{2}}, shape_data);
    {
        std::vector<Tensor*> inputs{&x, &shape_tensor};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(reshape_input.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{2, 6}}));
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], values[i]);
        }
    }

    // Transpose perm {1,0,2} on [2,3,2] -> [3,2,2].
    TransposeOp transpose;
    Attrs perm_attrs;
    perm_attrs["perm"] = std::vector<int64_t>{1, 0, 2};
    ASSERT_TRUE(transpose.configure(perm_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(transpose.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 2, 2}}));
        for (int64_t a = 0; a < 3; ++a) {
            for (int64_t b = 0; b < 2; ++b) {
                for (int64_t c = 0; c < 2; ++c) {
                    const std::size_t out_index = static_cast<std::size_t>(
                        (a * 2 + b) * 2 + c);
                    const std::size_t in_index = static_cast<std::size_t>(
                        (b * 3 + a) * 2 + c);
                    EXPECT_FLOAT_EQ(out[0].data<float>()[out_index],
                                    values[in_index]);
                }
            }
        }
    }

    // Transpose without perm reverses every axis.
    TransposeOp reverse;
    ASSERT_TRUE(reverse.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(reverse.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{2, 3, 2}}));
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t j = 0; j < 3; ++j) {
                for (int64_t k = 0; k < 2; ++k) {
                    const std::size_t out_index = static_cast<std::size_t>(
                        (i * 3 + j) * 2 + k);
                    const std::size_t in_index = static_cast<std::size_t>(
                        (k * 3 + j) * 2 + i);
                    EXPECT_FLOAT_EQ(out[0].data<float>()[out_index],
                                    values[in_index]);
                }
            }
        }
    }
}

TEST(TraditionalOpsTest, ReshapeInputShapeTakesPrecedenceOverAttribute) {
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    auto x = make_tensor(Shape{{2, 3, 2}}, values.data());
    const float valid_shape_data[2] = {3.0f, 4.0f};
    auto valid_shape = make_tensor(Shape{{2}}, valid_shape_data);
    ExecutionContext ctx = make_context();

    // The attribute alone would be invalid (81 elements); inputs[1] must win
    // and produce [3,4].
    ReshapeOp op;
    Attrs attrs;
    attrs["shape"] = std::vector<int64_t>{9, 9};
    ASSERT_TRUE(op.configure(attrs).has_value());
    {
        std::vector<Tensor*> inputs{&x, &valid_shape};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 4}}));
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(out[0].data<float>()[i], values[i]);
        }
    }

    // An invalid inputs[1] must fail even when the attribute is valid.
    ReshapeOp valid_attr_op;
    Attrs valid_attrs;
    valid_attrs["shape"] = std::vector<int64_t>{3, 4};
    ASSERT_TRUE(valid_attr_op.configure(valid_attrs).has_value());
    const float invalid_shape_data[2] = {5.0f, 5.0f};
    auto invalid_shape = make_tensor(Shape{{2}}, invalid_shape_data);
    {
        std::vector<Tensor*> inputs{&x, &invalid_shape};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_EQ(valid_attr_op.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);
    }
}

TEST(TraditionalOpsTest, LayoutOpsRejectInvalidArguments) {
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    auto x = make_tensor(Shape{{2, 3, 2}}, values.data());
    ExecutionContext ctx = make_context();
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    std::vector<Tensor*> inputs{&x};

    FlattenOp bad_axis;
    Attrs axis_attrs;
    axis_attrs["axis"] = static_cast<int64_t>(4);
    ASSERT_TRUE(bad_axis.configure(axis_attrs).has_value());
    EXPECT_EQ(bad_axis.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    FlattenOp bad_negative_axis;
    Attrs negative_axis;
    negative_axis["axis"] = static_cast<int64_t>(-4);
    ASSERT_TRUE(bad_negative_axis.configure(negative_axis).has_value());
    EXPECT_EQ(bad_negative_axis.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    ReshapeOp bad_count;
    Attrs count_attrs;
    count_attrs["shape"] = std::vector<int64_t>{5, 5};
    ASSERT_TRUE(bad_count.configure(count_attrs).has_value());
    EXPECT_EQ(bad_count.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    ReshapeOp missing_shape;
    ASSERT_TRUE(missing_shape.configure(Attrs{}).has_value());
    EXPECT_EQ(missing_shape.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    ReshapeOp two_inferred;
    Attrs inferred_attrs;
    inferred_attrs["shape"] = std::vector<int64_t>{-1, -1};
    ASSERT_TRUE(two_inferred.configure(inferred_attrs).has_value());
    EXPECT_EQ(two_inferred.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    TransposeOp wrong_perm_size;
    Attrs short_perm;
    short_perm["perm"] = std::vector<int64_t>{0, 1};
    ASSERT_TRUE(wrong_perm_size.configure(short_perm).has_value());
    EXPECT_EQ(wrong_perm_size.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    TransposeOp duplicate_perm;
    Attrs duplicate;
    duplicate["perm"] = std::vector<int64_t>{0, 0, 2};
    ASSERT_TRUE(duplicate_perm.configure(duplicate).has_value());
    EXPECT_EQ(duplicate_perm.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    TransposeOp out_of_range;
    Attrs range_attrs;
    range_attrs["perm"] = std::vector<int64_t>{0, 1, 3};
    ASSERT_TRUE(out_of_range.configure(range_attrs).has_value());
    EXPECT_EQ(out_of_range.forward(inputs, outputs, ctx).error().code,
              Error::Code::InvalidArgument);

    auto half = x;
    half.dtype = DType::F16;
    std::vector<Tensor*> half_inputs{&half};
    FlattenOp flatten;
    ASSERT_TRUE(flatten.configure(Attrs{}).has_value());
    EXPECT_EQ(flatten.forward(half_inputs, outputs, ctx).error().code,
              Error::Code::Unsupported);
}

TEST(TraditionalOpsTest, ConcatSlicePadGatherReduce) {
    ExecutionContext ctx = make_context();

    // Concat along a negative axis: only the concatenated dimension may vary.
    const float a_data[4] = {1, 2, 3, 4};
    const float b_data[2] = {5, 6};
    auto a = make_tensor(Shape{{2, 2}}, a_data);
    auto b = make_tensor(Shape{{1, 2}}, b_data);
    ConcatOp concat;
    Attrs concat_attrs;
    concat_attrs["axis"] = static_cast<int64_t>(-2);
    ASSERT_TRUE(concat.configure(concat_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&a, &b};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(concat.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 2}}));
        expect_values(out[0], {1, 2, 3, 4, 5, 6});
    }

    // Slice with attribute parameters on [3,4].
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    auto x = make_tensor(Shape{{3, 4}}, values.data());
    SliceOp slice_attrs;
    Attrs slice_attr_values;
    slice_attr_values["starts"] = std::vector<int64_t>{0, 1};
    slice_attr_values["ends"] = std::vector<int64_t>{3, 3};
    slice_attr_values["axes"] = std::vector<int64_t>{0, 1};
    slice_attr_values["steps"] = std::vector<int64_t>{1, 1};
    ASSERT_TRUE(slice_attrs.configure(slice_attr_values).has_value());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(slice_attrs.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 2}}));
        expect_values(out[0], {2, 3, 6, 7, 10, 11});
    }

    // Slice opset-10 inputs: F32 values truncate to int64, negative axis.
    const float starts_data[1] = {0.9f};
    const float ends_data[1] = {3.0f};
    const float axes_data[1] = {-1.0f};
    const float steps_data[1] = {2.0f};
    auto starts = make_tensor(Shape{{1}}, starts_data);
    auto ends = make_tensor(Shape{{1}}, ends_data);
    auto axes = make_tensor(Shape{{1}}, axes_data);
    auto steps = make_tensor(Shape{{1}}, steps_data);
    SliceOp slice_inputs;
    ASSERT_TRUE(slice_inputs.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&x, &starts, &ends, &axes, &steps};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(slice_inputs.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 2}}));
        expect_values(out[0], {1, 3, 5, 7, 9, 11});
    }

    // Pad with the ONNX [begins..., ends...] layout and a constant value.
    const float pad_data[4] = {1, 2, 3, 4};
    auto pad_input = make_tensor(Shape{{2, 2}}, pad_data);
    PadOp pad;
    Attrs pad_attrs;
    pad_attrs["pads"] = std::vector<int64_t>{1, 1, 0, 0};
    pad_attrs["value"] = 9.0;
    ASSERT_TRUE(pad.configure(pad_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&pad_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(pad.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 3}}));
        expect_values(out[0], {9, 9, 9, 9, 1, 2, 9, 3, 4});
    }

    // Pad opset-11 inputs: pads and constant value tensors.
    const float one_d_data[3] = {1, 2, 3};
    auto one_d = make_tensor(Shape{{3}}, one_d_data);
    const float pads_data[2] = {0.0f, 2.0f};
    const float value_data[1] = {7.0f};
    auto pads_tensor = make_tensor(Shape{{2}}, pads_data);
    auto value_tensor = make_tensor(Shape{{1}}, value_data);
    PadOp pad_inputs;
    ASSERT_TRUE(pad_inputs.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&one_d, &pads_tensor, &value_tensor};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(pad_inputs.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{5}}));
        expect_values(out[0], {1, 2, 3, 7, 7});
    }

    const float gather_data[12] = {1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12};
    auto gather_input = make_tensor(Shape{{3, 4}}, gather_data);
    const float index_data[2] = {2.0f, 0.0f};
    auto indices = make_tensor(Shape{{2}}, index_data);

    // Gather with a negative axis.
    GatherOp gather;
    Attrs gather_attrs;
    gather_attrs["axis"] = static_cast<int64_t>(-1);
    ASSERT_TRUE(gather.configure(gather_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&gather_input, &indices};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(gather.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 2}}));
        expect_values(out[0], {3, 1, 7, 5, 11, 9});
    }

    // Gather normalizes negative indices.
    const float negative_index_data[1] = {-1.0f};
    auto negative_index = make_tensor(Shape{{1}}, negative_index_data);
    GatherOp gather_last;
    Attrs gather_axis_zero;
    gather_axis_zero["axis"] = static_cast<int64_t>(0);
    ASSERT_TRUE(gather_last.configure(gather_axis_zero).has_value());
    {
        std::vector<Tensor*> inputs{&gather_input, &negative_index};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(gather_last.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 4}}));
        expect_values(out[0], {9, 10, 11, 12});
    }

    // ReduceMean over axis 1 with keepdims.
    ReduceOp mean;
    Attrs mean_attrs;
    mean_attrs["axes"] = std::vector<int64_t>{1};
    mean_attrs["keepdims"] = static_cast<int64_t>(1);
    mean_attrs["reduce_mode"] = std::string("mean");
    ASSERT_TRUE(mean.configure(mean_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&gather_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(mean.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3, 1}}));
        expect_values(out[0], {2.5f, 6.5f, 10.5f});
    }

    // ReduceSum over axis 0 without keepdims.
    ReduceOp sum;
    Attrs sum_attrs;
    sum_attrs["axes"] = std::vector<int64_t>{0};
    sum_attrs["keepdims"] = static_cast<int64_t>(0);
    sum_attrs["reduce_mode"] = std::string("sum");
    ASSERT_TRUE(sum.configure(sum_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&gather_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(sum.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{4}}));
        expect_values(out[0], {15, 18, 21, 24});
    }

    // An empty axes list reduces over every axis; keepdims defaults to 1.
    ReduceOp mean_all;
    Attrs mean_all_attrs;
    mean_all_attrs["reduce_mode"] = std::string("mean");
    ASSERT_TRUE(mean_all.configure(mean_all_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&gather_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(mean_all.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1}}));
        expect_values(out[0], {6.5f});
    }
}

TEST(TraditionalOpsTest, UpsampleNearestAndClipSigmoidTanh) {
    ExecutionContext ctx = make_context();

    // Nearest-neighbor NCHW upsample from the scales attribute.
    const float upsample_data[4] = {1, 2, 3, 4};
    auto upsample_input = make_tensor(Shape{{1, 1, 2, 2}}, upsample_data);
    UpsampleOp upsample;
    Attrs upsample_attrs;
    upsample_attrs["scales"] = std::vector<double>{1.0, 1.0, 2.0, 2.0};
    upsample_attrs["mode"] = std::string("nearest");
    ASSERT_TRUE(upsample.configure(upsample_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&upsample_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(upsample.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1, 4, 4}}));
        expect_values(out[0], {1, 1, 2, 2, 1, 1, 2, 2,
                               3, 3, 4, 4, 3, 3, 4, 4});
    }

    // Upsample from an F32 scales input tensor.
    const float row_data[2] = {5, 6};
    auto row_input = make_tensor(Shape{{1, 1, 1, 2}}, row_data);
    const float scale_data[4] = {1.0f, 1.0f, 1.0f, 2.0f};
    auto scale_tensor = make_tensor(Shape{{4}}, scale_data);
    UpsampleOp upsample_inputs;
    ASSERT_TRUE(upsample_inputs.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&row_input, &scale_tensor};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(upsample_inputs.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{1, 1, 1, 4}}));
        expect_values(out[0], {5, 5, 6, 6});
    }

    // Clip with attribute bounds.
    const float clip_data[5] = {-2, -1, 0, 1, 2};
    auto clip_input = make_tensor(Shape{{5}}, clip_data);
    ClipOp clip;
    Attrs clip_attrs;
    clip_attrs["min"] = -1.0;
    clip_attrs["max"] = 1.0;
    ASSERT_TRUE(clip.configure(clip_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&clip_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(clip.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{5}}));
        expect_values(out[0], {-1, -1, 0, 1, 1});
    }

    // Clip with scalar min/max input tensors.
    const float min_data[1] = {0.5f};
    const float max_data[1] = {1.5f};
    auto min_tensor = make_tensor(Shape{{1}}, min_data);
    auto max_tensor = make_tensor(Shape{{1}}, max_data);
    ClipOp clip_inputs;
    ASSERT_TRUE(clip_inputs.configure(Attrs{}).has_value());
    {
        std::vector<Tensor*> inputs{&clip_input, &min_tensor, &max_tensor};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(clip_inputs.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{5}}));
        expect_values(out[0], {0.5f, 0.5f, 0.5f, 1.0f, 1.5f});
    }

    // A missing upper bound leaves positive values untouched.
    ClipOp clip_lower;
    Attrs lower_attrs;
    lower_attrs["min"] = 0.0;
    ASSERT_TRUE(clip_lower.configure(lower_attrs).has_value());
    {
        std::vector<Tensor*> inputs{&clip_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(clip_lower.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{5}}));
        expect_values(out[0], {0, 0, 0, 1, 2});
    }

    // Sigmoid is elementwise and stays strictly inside (0, 1).
    const float sigmoid_data[4] = {-2, -1, 0, 2};
    auto sigmoid_input = make_tensor(Shape{{4}}, sigmoid_data);
    SigmoidOp sigmoid;
    {
        std::vector<Tensor*> inputs{&sigmoid_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(sigmoid.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{4}}));
        expect_values(out[0], {0.11920292f, 0.26894143f, 0.5f, 0.88079708f},
                      1e-6f);
        for (int64_t i = 0; i < sigmoid_input.shape.numel(); ++i) {
            EXPECT_GT(out[0].data<float>()[i], 0.0f);
            EXPECT_LT(out[0].data<float>()[i], 1.0f);
        }
    }

    // Tanh is elementwise and its magnitude stays strictly below 1.
    const float tanh_data[3] = {-1, 0, 1};
    auto tanh_input = make_tensor(Shape{{3}}, tanh_data);
    TanhOp tanh;
    {
        std::vector<Tensor*> inputs{&tanh_input};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        ASSERT_TRUE(tanh.forward(inputs, outputs, ctx).has_value());
        ASSERT_EQ(out[0].shape, (Shape{{3}}));
        expect_values(out[0], {-0.76159416f, 0.0f, 0.76159416f}, 1e-6f);
        for (int64_t i = 0; i < tanh_input.shape.numel(); ++i) {
            EXPECT_LT(std::fabs(out[0].data<float>()[i]), 1.0f);
        }
    }
}

TEST(TraditionalOpsTest, SliceHandlesNegativeEndsAndSteps) {
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    auto x = make_tensor(Shape{{3, 4}}, values.data());
    ExecutionContext ctx = make_context();

    const auto run = [&](const Attrs& attrs) {
        SliceOp op;
        EXPECT_TRUE(op.configure(attrs).has_value());
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_TRUE(op.forward(inputs, outputs, ctx).has_value());
        return out[0];
    };

    // Negative start/end with a negative step reverses every row.
    Attrs reverse_rows;
    reverse_rows["starts"] = std::vector<int64_t>{-1};
    reverse_rows["ends"] = std::vector<int64_t>{-5};
    reverse_rows["axes"] = std::vector<int64_t>{-1};
    reverse_rows["steps"] = std::vector<int64_t>{-1};
    Tensor reversed = run(reverse_rows);
    ASSERT_EQ(reversed.shape, (Shape{{3, 4}}));
    expect_values(reversed, {4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9});

    // Stride 2 over rows while stepping backwards.
    Attrs strided_rows;
    strided_rows["starts"] = std::vector<int64_t>{2};
    strided_rows["ends"] = std::vector<int64_t>{-4};
    strided_rows["axes"] = std::vector<int64_t>{0};
    strided_rows["steps"] = std::vector<int64_t>{-2};
    Tensor strided = run(strided_rows);
    ASSERT_EQ(strided.shape, (Shape{{2, 4}}));
    expect_values(strided, {9, 10, 11, 12, 1, 2, 3, 4});

    // Negative indices with a positive step on the last axis.
    Attrs negative_indices;
    negative_indices["starts"] = std::vector<int64_t>{-3};
    negative_indices["ends"] = std::vector<int64_t>{-1};
    negative_indices["axes"] = std::vector<int64_t>{-1};
    Tensor middle = run(negative_indices);
    ASSERT_EQ(middle.shape, (Shape{{3, 2}}));
    expect_values(middle, {2, 3, 6, 7, 10, 11});

    // A very negative end means "slice to the beginning" for negative steps.
    Attrs to_beginning;
    to_beginning["starts"] = std::vector<int64_t>{-1};
    to_beginning["ends"] = std::vector<int64_t>{-2147483648};
    to_beginning["axes"] = std::vector<int64_t>{0};
    to_beginning["steps"] = std::vector<int64_t>{-1};
    Tensor beginning = run(to_beginning);
    ASSERT_EQ(beginning.shape, (Shape{{3, 4}}));
    expect_values(beginning, {9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4});
}

TEST(TraditionalOpsTest, DataAndActivationOpsRejectInvalidArguments) {
    ExecutionContext ctx = make_context();

    // Concat requires equal non-axis dimensions and an in-range axis.
    const float a_data[4] = {1, 2, 3, 4};
    const float c_data[2] = {5, 6};
    auto a = make_tensor(Shape{{2, 2}}, a_data);
    auto c = make_tensor(Shape{{2, 1}}, c_data);
    ConcatOp concat;
    Attrs concat_axis_zero;
    concat_axis_zero["axis"] = static_cast<int64_t>(0);
    ASSERT_TRUE(concat.configure(concat_axis_zero).has_value());
    {
        std::vector<Tensor*> inputs{&a, &c};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_EQ(concat.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);
    }
    ConcatOp concat_bad_axis;
    Attrs concat_axis_two;
    concat_axis_two["axis"] = static_cast<int64_t>(2);
    ASSERT_TRUE(concat_bad_axis.configure(concat_axis_two).has_value());
    {
        std::vector<Tensor*> inputs{&a, &c};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};
        EXPECT_EQ(concat_bad_axis.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);
    }

    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    auto x = make_tensor(Shape{{3, 4}}, values.data());
    {
        std::vector<Tensor*> inputs{&x};
        std::vector<Tensor> out(1);
        std::vector<Tensor*> outputs{&out[0]};

        // Slice rejects zero steps, out-of-range axes, and empty outputs.
        SliceOp zero_step;
        Attrs zero_step_attrs;
        zero_step_attrs["starts"] = std::vector<int64_t>{0};
        zero_step_attrs["ends"] = std::vector<int64_t>{3};
        zero_step_attrs["steps"] = std::vector<int64_t>{0};
        ASSERT_TRUE(zero_step.configure(zero_step_attrs).has_value());
        EXPECT_EQ(zero_step.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        SliceOp bad_axis;
        Attrs bad_axis_attrs;
        bad_axis_attrs["starts"] = std::vector<int64_t>{0};
        bad_axis_attrs["ends"] = std::vector<int64_t>{1};
        bad_axis_attrs["axes"] = std::vector<int64_t>{2};
        ASSERT_TRUE(bad_axis.configure(bad_axis_attrs).has_value());
        EXPECT_EQ(bad_axis.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        SliceOp empty_result;
        Attrs empty_attrs;
        empty_attrs["starts"] = std::vector<int64_t>{3};
        empty_attrs["ends"] = std::vector<int64_t>{3};
        empty_attrs["axes"] = std::vector<int64_t>{0};
        ASSERT_TRUE(empty_result.configure(empty_attrs).has_value());
        EXPECT_EQ(empty_result.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        // Pad rejects malformed pads and non-constant modes.
        PadOp bad_pads;
        Attrs bad_pads_attrs;
        bad_pads_attrs["pads"] = std::vector<int64_t>{1, 1, 1};
        ASSERT_TRUE(bad_pads.configure(bad_pads_attrs).has_value());
        EXPECT_EQ(bad_pads.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        PadOp cropped_empty;
        Attrs cropped_attrs;
        cropped_attrs["pads"] = std::vector<int64_t>{-4, 0};
        ASSERT_TRUE(cropped_empty.configure(cropped_attrs).has_value());
        EXPECT_EQ(cropped_empty.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        PadOp reflect;
        Attrs reflect_attrs;
        reflect_attrs["mode"] = std::string("reflect");
        EXPECT_EQ(reflect.configure(reflect_attrs).error().code,
                  Error::Code::Unsupported);

        // Upsample only supports NCHW nearest with positive scales.
        const float small_data[6] = {1, 2, 3, 4, 5, 6};
        auto rank_three = make_tensor(Shape{{1, 2, 3}}, small_data);
        UpsampleOp rank_three_op;
        Attrs rank_three_attrs;
        rank_three_attrs["scales"] = std::vector<double>{1.0, 2.0, 2.0};
        ASSERT_TRUE(rank_three_op.configure(rank_three_attrs).has_value());
        std::vector<Tensor*> rank_three_inputs{&rank_three};
        EXPECT_EQ(rank_three_op.forward(rank_three_inputs, outputs, ctx)
                      .error()
                      .code,
                  Error::Code::InvalidArgument);

        UpsampleOp zero_scale_op;
        Attrs zero_scale_attrs;
        zero_scale_attrs["scales"] =
            std::vector<double>{1.0, 1.0, 0.0, 2.0};
        ASSERT_TRUE(zero_scale_op.configure(zero_scale_attrs).has_value());
        auto image = make_tensor(Shape{{1, 1, 2, 2}}, a_data);
        std::vector<Tensor*> image_inputs2{&image};
        EXPECT_EQ(zero_scale_op.forward(image_inputs2, outputs, ctx)
                      .error()
                      .code,
                  Error::Code::InvalidArgument);

        UpsampleOp linear;
        Attrs linear_attrs;
        linear_attrs["mode"] = std::string("linear");
        EXPECT_EQ(linear.configure(linear_attrs).error().code,
                  Error::Code::Unsupported);

        // Gather normalizes but rejects out-of-range indices and axes.
        const float index_data[1] = {4.0f};
        auto oob_index = make_tensor(Shape{{1}}, index_data);
        GatherOp gather;
        ASSERT_TRUE(gather.configure(Attrs{}).has_value());
        std::vector<Tensor*> gather_inputs{&x, &oob_index};
        EXPECT_EQ(gather.forward(gather_inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        GatherOp gather_bad_axis;
        Attrs gather_axis_attrs;
        gather_axis_attrs["axis"] = static_cast<int64_t>(2);
        ASSERT_TRUE(gather_bad_axis.configure(gather_axis_attrs).has_value());
        const float good_index_data[1] = {0.0f};
        auto good_index = make_tensor(Shape{{1}}, good_index_data);
        std::vector<Tensor*> gather_axis_inputs{&x, &good_index};
        EXPECT_EQ(gather_bad_axis.forward(gather_axis_inputs, outputs, ctx)
                      .error()
                      .code,
                  Error::Code::InvalidArgument);

        // Reduce rejects unknown modes and duplicate axes.
        ReduceOp bad_mode;
        Attrs bad_mode_attrs;
        bad_mode_attrs["reduce_mode"] = std::string("max");
        EXPECT_EQ(bad_mode.configure(bad_mode_attrs).error().code,
                  Error::Code::InvalidArgument);

        ReduceOp duplicate_axes;
        Attrs duplicate_attrs;
        duplicate_attrs["axes"] = std::vector<int64_t>{1, -1};
        duplicate_attrs["reduce_mode"] = std::string("sum");
        ASSERT_TRUE(duplicate_axes.configure(duplicate_attrs).has_value());
        EXPECT_EQ(duplicate_axes.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        // Clip rejects min > max and too many inputs.
        ClipOp inverted;
        Attrs inverted_attrs;
        inverted_attrs["min"] = 2.0;
        inverted_attrs["max"] = 1.0;
        ASSERT_TRUE(inverted.configure(inverted_attrs).has_value());
        EXPECT_EQ(inverted.forward(inputs, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        const float clip_data[2] = {-1, 1};
        auto clip_input = make_tensor(Shape{{2}}, clip_data);
        const float bound_data[1] = {0.0f};
        auto bound = make_tensor(Shape{{1}}, bound_data);
        ClipOp extra_inputs;
        ASSERT_TRUE(extra_inputs.configure(Attrs{}).has_value());
        std::vector<Tensor*> clip_many{&clip_input, &bound, &bound, &bound};
        EXPECT_EQ(extra_inputs.forward(clip_many, outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        // Elementwise activations validate dtype/count/allocator.
        auto half = clip_input;
        half.dtype = DType::F16;
        SigmoidOp sigmoid;
        std::vector<Tensor*> half_inputs{&half};
        EXPECT_EQ(sigmoid.forward(half_inputs, outputs, ctx).error().code,
                  Error::Code::Unsupported);

        TanhOp tanh;
        std::vector<Tensor*> no_outputs;
        EXPECT_EQ(tanh.forward(inputs, no_outputs, ctx).error().code,
                  Error::Code::InvalidArgument);

        ExecutionContext no_allocator;
        std::vector<Tensor> tanh_out(1);
        std::vector<Tensor*> tanh_outputs{&tanh_out[0]};
        EXPECT_EQ(tanh.forward(inputs, tanh_outputs, no_allocator).error().code,
                  Error::Code::InvalidArgument);
    }
}

TEST(TraditionalOpsTest, RegistryCreatesDataAndActivationOps) {
    auto& registry = OpRegistry::instance();
    register_builtin_ops(registry);

    for (OpType type :
         {OpType::Concat, OpType::Slice, OpType::Pad, OpType::Upsample,
          OpType::Gather, OpType::Reduce, OpType::Clip, OpType::Sigmoid,
          OpType::Tanh}) {
        auto op = registry.create(type);
        ASSERT_TRUE(op.has_value());
        EXPECT_NE(op->get(), nullptr);
    }
    auto concat = registry.create(OpType::Concat);
    ASSERT_TRUE(concat.has_value());
    EXPECT_NE(dynamic_cast<ConcatOp*>(concat->get()), nullptr);
    auto upsample = registry.create(OpType::Upsample);
    ASSERT_TRUE(upsample.has_value());
    EXPECT_NE(dynamic_cast<UpsampleOp*>(upsample->get()), nullptr);
    auto sigmoid = registry.create(OpType::Sigmoid);
    ASSERT_TRUE(sigmoid.has_value());
    EXPECT_NE(dynamic_cast<SigmoidOp*>(sigmoid->get()), nullptr);
    auto tanh = registry.create(OpType::Tanh);
    ASSERT_TRUE(tanh.has_value());
    EXPECT_NE(dynamic_cast<TanhOp*>(tanh->get()), nullptr);
}

TEST(TraditionalOpsTest, RegistryCreatesTraditionalOps) {
    auto& registry = OpRegistry::instance();
    register_builtin_ops(registry);

    for (OpType type :
         {OpType::Conv2D, OpType::DepthwiseConv2D, OpType::MaxPool2D,
          OpType::AvgPool2D, OpType::GlobalAvgPool, OpType::BatchNorm,
          OpType::Gemm, OpType::Flatten, OpType::Reshape, OpType::Transpose}) {
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
    auto gemm = registry.create(OpType::Gemm);
    ASSERT_TRUE(gemm.has_value());
    EXPECT_NE(dynamic_cast<GemmOp*>(gemm->get()), nullptr);
    auto flatten = registry.create(OpType::Flatten);
    ASSERT_TRUE(flatten.has_value());
    EXPECT_NE(dynamic_cast<FlattenOp*>(flatten->get()), nullptr);
    auto reshape = registry.create(OpType::Reshape);
    ASSERT_TRUE(reshape.has_value());
    EXPECT_NE(dynamic_cast<ReshapeOp*>(reshape->get()), nullptr);
    auto transpose = registry.create(OpType::Transpose);
    ASSERT_TRUE(transpose.has_value());
    EXPECT_NE(dynamic_cast<TransposeOp*>(transpose->get()), nullptr);
}
