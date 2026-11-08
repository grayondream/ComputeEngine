#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/memory/malloc_allocator.h"
#include "core/ops/common/matmul_op.h"
#include "core/ops/common/relu_op.h"
#include "core/ops/common/softmax_op.h"
#include "core/runtime/executor.h"
#include "core/runtime/thread_pool.h"

using namespace ie;

namespace {

std::atomic<int> g_matmul_calls{0};
void fake_matmul_row(const float* /*a_row*/, const float* /*b*/, float* out,
                     std::int64_t /*k*/, std::int64_t n) {
    ++g_matmul_calls;
    for (std::int64_t j = 0; j < n; ++j) {
        out[j] = 1.0f;
    }
}

std::atomic<int> g_relu_calls{0};
void fake_relu(const float* in, float* out, std::int64_t n) {
    ++g_relu_calls;
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = in[i] + 100.0f;
    }
}

std::atomic<int> g_softmax_calls{0};
void fake_softmax(const float* /*in*/, float* out, std::int64_t rows,
                  std::int64_t cols) {
    ++g_softmax_calls;
    const float value = 1.0f / static_cast<float>(rows * cols);
    for (std::int64_t i = 0; i < rows * cols; ++i) {
        out[i] = value;
    }
}

Status run_forward(Op& op, std::initializer_list<Tensor*> inputs,
                   std::initializer_list<Tensor*> outputs,
                   ExecutionContext& context) {
    std::vector<Tensor*> in(inputs);
    std::vector<Tensor*> out(outputs);
    return op.forward(in, out, context);
}

}  // namespace

TEST(ReluOpTest, AppliesElementwiseAndValidates) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ReluOp op;
    Tensor in = Tensor::from_data({-2.0f, -0.0f, 0.0f, 3.0f});
    Tensor out;
    std::vector<Tensor*> ins{&in}, outs{&out};
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[2], 0.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 3.0f);
    ASSERT_EQ(out.shape.dims, (std::vector<int64_t>{4}));

    std::vector<Tensor*> empty;
    EXPECT_EQ(op.forward(empty, outs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(ReluOpTest, UsesInjectedKernelAndValidatesInputs) {
    auto alloc = std::make_shared<MallocAllocator>();
    KernelFns fns;
    fns.relu = fake_relu;
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ctx.kernels = &fns;
    ReluOp op;
    Tensor in = Tensor::from_data({-1.0f, 0.0f, 2.0f});
    Tensor out;
    g_relu_calls = 0;
    ASSERT_TRUE(run_forward(op, {&in}, {&out}, ctx).has_value());
    EXPECT_EQ(g_relu_calls.load(), 1);
    EXPECT_FLOAT_EQ(out.data<float>()[0], 99.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[2], 102.0f);

    ExecutionContext no_alloc;
    EXPECT_EQ(run_forward(op, {&in}, {&out}, no_alloc).error().code,
              Error::Code::InvalidArgument);

    Tensor bad = Tensor::from_data({1.0f});
    bad.dtype = DType::F16;
    ExecutionContext ok;
    ok.allocator = alloc;
    EXPECT_EQ(run_forward(op, {&bad}, {&out}, ok).error().code,
              Error::Code::Unsupported);

    EXPECT_EQ(run_forward(op, {&in}, {&out, &out}, ok).error().code,
              Error::Code::InvalidArgument);
}

TEST(SoftmaxOpTest, NormalizesLastAxisAndIsStable) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    SoftmaxOp op;
    Tensor in = Tensor::from_data({1000.0f, 1000.0f, 1000.0f, 1000.0f});
    in.shape = Shape{{2, 2}};
    Tensor out;
    std::vector<Tensor*> ins{&in}, outs{&out};
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_NEAR(out.data<float>()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(out.data<float>()[2], 0.5f, 1e-6f);
    EXPECT_TRUE(std::isfinite(out.data<float>()[0]));
}

TEST(SoftmaxOpTest, HandlesOneDimensionalInput) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    SoftmaxOp op;
    Tensor in = Tensor::from_data({1.0f, 2.0f, 3.0f});
    Tensor out;
    ASSERT_TRUE(run_forward(op, {&in}, {&out}, ctx).has_value());
    ASSERT_EQ(out.shape.dims, (std::vector<int64_t>{3}));

    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += out.data<float>()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    EXPECT_GT(out.data<float>()[2], out.data<float>()[1]);
    EXPECT_GT(out.data<float>()[1], out.data<float>()[0]);
}

TEST(SoftmaxOpTest, NormalizesEachTwoDimensionalRow) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    SoftmaxOp op;
    Tensor in = Tensor::from_data(
        {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.0f});
    in.shape = Shape{{2, 3}};
    Tensor out;
    ASSERT_TRUE(run_forward(op, {&in}, {&out}, ctx).has_value());
    ASSERT_EQ(out.shape.dims, (std::vector<int64_t>{2, 3}));
    for (int64_t r = 0; r < 2; ++r) {
        float sum = 0.0f;
        for (int64_t c = 0; c < 3; ++c) {
            const float v = out.data<float>()[r * 3 + c];
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, 0.0f);
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST(SoftmaxOpTest, UsesInjectedKernelAndRejectsUnsupportedDims) {
    auto alloc = std::make_shared<MallocAllocator>();
    KernelFns fns;
    fns.softmax = fake_softmax;
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ctx.kernels = &fns;
    SoftmaxOp op;

    Tensor in = Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f});
    in.shape = Shape{{2, 2}};
    Tensor out;
    g_softmax_calls = 0;
    ASSERT_TRUE(run_forward(op, {&in}, {&out}, ctx).has_value());
    EXPECT_EQ(g_softmax_calls.load(), 1);
    EXPECT_NEAR(out.data<float>()[0], 0.25f, 1e-6f);
    EXPECT_NEAR(out.data<float>()[3], 0.25f, 1e-6f);

    Tensor ranked = Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f,
                                       5.0f, 6.0f, 7.0f, 8.0f});
    ranked.shape = Shape{{2, 2, 2}};
    EXPECT_EQ(run_forward(op, {&ranked}, {&out}, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(MatMulOpTest, UsesInjectedKernelAndThreadsRows) {
    auto alloc = std::make_shared<MallocAllocator>();
    ThreadPool pool(4);
    KernelFns fns;
    fns.matmul_row = fake_matmul_row;
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ctx.kernels = &fns;
    ctx.thread_pool = &pool;

    auto a_alloc = Tensor::allocate(Shape{{64, 64}}, DType::F32, alloc);
    ASSERT_TRUE(a_alloc.has_value());
    std::fill(a_alloc->data<float>(), a_alloc->data<float>() + 64 * 64, 1.0f);
    Tensor a = std::move(*a_alloc);
    Tensor b = a;
    Tensor out;
    std::vector<Tensor*> ins{&a, &b}, outs{&out};
    g_matmul_calls = 0;
    MatMulOp op;
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_EQ(g_matmul_calls.load(), 64);
    EXPECT_FLOAT_EQ(out.data<float>()[0], 1.0f);
}

TEST(MatMulOpTest, ThreadedAndSerialResultsAreIdentical) {
    auto alloc = std::make_shared<MallocAllocator>();
    constexpr int64_t m = 64;
    constexpr int64_t k = 64;
    constexpr int64_t n = 64;

    auto a_alloc = Tensor::allocate(Shape{{m, k}}, DType::F32, alloc);
    auto b_alloc = Tensor::allocate(Shape{{k, n}}, DType::F32, alloc);
    ASSERT_TRUE(a_alloc.has_value());
    ASSERT_TRUE(b_alloc.has_value());
    Tensor a = std::move(*a_alloc);
    Tensor b = std::move(*b_alloc);
    std::uint32_t state = 12345;
    const auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xffff) / 65535.0f - 0.5f;
    };
    for (int64_t i = 0; i < m * k; ++i) {
        a.data<float>()[i] = next();
    }
    for (int64_t i = 0; i < k * n; ++i) {
        b.data<float>()[i] = next();
    }

    MatMulOp op;
    ExecutionContext serial_ctx;
    serial_ctx.allocator = alloc;
    Tensor serial;
    ASSERT_TRUE(
        run_forward(op, {&a, &b}, {&serial}, serial_ctx).has_value());

    ThreadPool pool(4);
    ExecutionContext threaded_ctx = serial_ctx;
    threaded_ctx.thread_pool = &pool;
    Tensor threaded;
    ASSERT_TRUE(
        run_forward(op, {&a, &b}, {&threaded}, threaded_ctx).has_value());

    ASSERT_EQ(serial.shape.dims, (std::vector<int64_t>{m, n}));
    ASSERT_EQ(threaded.shape.dims, (std::vector<int64_t>{m, n}));
    EXPECT_EQ(std::memcmp(serial.data<float>(), threaded.data<float>(),
                          static_cast<std::size_t>(m * n) * sizeof(float)),
              0);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                expected += a.data<float>()[i * k + t] *
                            b.data<float>()[t * n + j];
            }
            EXPECT_FLOAT_EQ(threaded.data<float>()[i * n + j], expected);
        }
    }
}

TEST(MatMulOpTest, SmallProductsStaySerialAndMatchReference) {
    auto alloc = std::make_shared<MallocAllocator>();
    ThreadPool pool(4);
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ctx.thread_pool = &pool;

    Tensor a = Tensor::from_data({1.0f, 2.0f, 3.0f, 4.0f});
    a.shape = Shape{{2, 2}};
    Tensor b = Tensor::from_data({5.0f, 6.0f, 7.0f, 8.0f});
    b.shape = Shape{{2, 2}};
    Tensor out;
    g_matmul_calls = 0;
    MatMulOp op;
    ASSERT_TRUE(run_forward(op, {&a, &b}, {&out}, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 19.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 50.0f);
}

TEST(ExecutorTest, InjectsKernelsAndKeepsDefaultArguments) {
    auto alloc = std::make_shared<MallocAllocator>();
    Graph graph;
    graph.values.push_back(Value{0, "a", Shape{{2}}, DType::F32, true, false});
    graph.values.push_back(
        Value{1, "out", Shape{{2}}, DType::F32, false, true});
    Node node;
    node.id = 0;
    node.op = OpType::Relu;
    node.inputs = {0};
    node.outputs = {1};
    graph.nodes.push_back(node);

    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({-1.0f, 2.0f}));

    Executor executor;
    auto scalar = executor.run(graph, inputs, alloc);
    ASSERT_TRUE(scalar.has_value());
    ASSERT_EQ(scalar->size(), 1u);
    EXPECT_FLOAT_EQ(scalar->at(0).data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(scalar->at(0).data<float>()[1], 2.0f);

    KernelFns fns;
    fns.relu = fake_relu;
    g_relu_calls = 0;
    auto dispatched = executor.run(graph, inputs, alloc, &fns);
    ASSERT_TRUE(dispatched.has_value());
    EXPECT_EQ(g_relu_calls.load(), 1);
    EXPECT_FLOAT_EQ(dispatched->at(0).data<float>()[0], 99.0f);
    EXPECT_FLOAT_EQ(dispatched->at(0).data<float>()[1], 102.0f);
}
