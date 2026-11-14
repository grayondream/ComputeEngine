#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "core/ir/graph_builder.h"
#include "core/memory/malloc_allocator.h"
#include "core/runtime/executor.h"

using namespace ie;

TEST(GraphBuilderTest, BuildsAddThenScalarMul) {
    GraphBuilder b;
    Graph g = std::move(b).add(2).mul(3.0).build();

    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].op, OpType::Add);
    EXPECT_EQ(g.nodes[0].inputs.size(), 2u);
    EXPECT_EQ(g.nodes[1].op, OpType::Mul);
    ASSERT_TRUE(std::holds_alternative<double>(g.nodes[1].attrs.at("scalar")));
    EXPECT_DOUBLE_EQ(std::get<double>(g.nodes[1].attrs.at("scalar")), 3.0);

    ASSERT_NE(g.input("a"), nullptr);
    ASSERT_NE(g.input("b"), nullptr);
    EXPECT_TRUE(g.input("a")->is_input);
    EXPECT_EQ(g.nodes[1].outputs.size(), 1u);
    const Value* out = g.find_value(g.nodes[1].outputs[0]);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->is_output);
}

TEST(GraphTest, BindRequiresEveryInput) {
    GraphBuilder b;
    Graph g = std::move(b).add(2).mul(3.0).build();

    auto missing = g.bind(ShapeMap{{"a", Shape{{3}}}});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, Error::Code::InvalidArgument);

    auto ok = g.bind(ShapeMap{{"a", Shape{{3}}}, {"b", Shape{{3}}}});
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->shapes.size(), 2u);
}

TEST(GraphTest, EmptyGraphIsDetectable) {
    Graph g;
    EXPECT_FALSE(g.is_dynamic());
    EXPECT_FALSE(g.has_control_flow());
    EXPECT_FALSE(g.has_backward());
}

TEST(GraphWeightsTest, ExecutorSeedsConstantWeights) {
    auto alloc = std::make_shared<MallocAllocator>();

    Graph graph;
    Value x; x.id = 0; x.name = "x"; x.shape = Shape{{2}}; x.is_input = true;
    Value w; w.id = 1; w.name = "w"; w.shape = Shape{{2}};
    Value y; y.id = 2; y.name = "y"; y.shape = Shape{{2}}; y.is_output = true;
    graph.values = {x, w, y};

    Node node; node.id = 0; node.op = OpType::Add; node.inputs = {0, 1};
    node.outputs = {2}; graph.nodes.push_back(node);

    auto weight = Tensor::allocate(Shape{{2}}, DType::F32, alloc,
                                   MemoryKind::WEIGHT_CONST);
    ASSERT_TRUE(weight.has_value());
    weight->data<float>()[0] = 10.0f;
    weight->data<float>()[1] = 20.0f;
    graph.weights.emplace("w", std::move(*weight));

    auto input = Tensor::allocate(Shape{{2}}, DType::F32, alloc);
    ASSERT_TRUE(input.has_value());
    input->data<float>()[0] = 1.0f;
    input->data<float>()[1] = 2.0f;

    Executor executor;
    auto out = executor.run(graph, {{"x", std::move(*input)}}, alloc);
    ASSERT_TRUE(out.has_value());
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[0], 11.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[1], 22.0f);
}

TEST(GraphWeightsTest, MissingWeightIsAnError) {
    auto alloc = std::make_shared<MallocAllocator>();
    Graph graph;
    Value x; x.id = 0; x.name = "x"; x.shape = Shape{{1}}; x.is_input = true;
    Value w; w.id = 1; w.name = "w"; w.shape = Shape{{1}};
    Value y; y.id = 2; y.name = "y"; y.shape = Shape{{1}}; y.is_output = true;
    graph.values = {x, w, y};
    Node node; node.id = 0; node.op = OpType::Add; node.inputs = {0, 1};
    node.outputs = {2}; graph.nodes.push_back(node);

    auto input = Tensor::allocate(Shape{{1}}, DType::F32, alloc);
    ASSERT_TRUE(input.has_value());
    input->data<float>()[0] = 1.0f;

    Executor executor;
    auto out = executor.run(graph, {{"x", std::move(*input)}}, alloc);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}
