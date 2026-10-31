#include <gtest/gtest.h>

#include "core/ir/graph_builder.h"
#include "core/runtime/engine.h"

using namespace ie;

static Graph BuildMulAddGraph() {
    GraphBuilder b;
    return std::move(b).add(2).mul(3.0).build();
}

static Graph BuildBinaryMulGraph() {
    Graph graph;
    graph.values.push_back(Value{0, "a", Shape{{3}}, DType::F32, true, false});
    graph.values.push_back(Value{1, "b", Shape{{3}}, DType::F32, true, false});
    graph.values.push_back(Value{2, "out", Shape{{3}}, DType::F32, false, true});
    Node node;
    node.id = 0;
    node.op = OpType::Mul;
    node.inputs = {0, 1};
    node.outputs = {2};
    graph.nodes.push_back(node);
    return graph;
}

TEST(EngineE2ETest, ComputesAddThenMul) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildMulAddGraph()).has_value());

    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    inputs.emplace("b", Tensor::from_data({4.0f, 5.0f, 6.0f}));

    auto out = engine.infer(inputs);
    ASSERT_TRUE(out.has_value());
    ASSERT_EQ(out->size(), 1u);
    ASSERT_EQ(out->at(0).shape.numel(), 3);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[0], 15.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[1], 21.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[2], 27.0f);
}

TEST(EngineE2ETest, MissingInputIsReported) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildMulAddGraph()).has_value());
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    auto out = engine.infer(inputs);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}

TEST(EngineE2ETest, InferBeforeLoadIsRejected) {
    EngineConfig cfg;
    Engine engine(cfg);
    auto out = engine.infer({});
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}

TEST(EngineE2ETest, GraphWithoutOutputsIsRejected) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(Graph{}).has_value());
    auto out = engine.infer({});
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}

TEST(EngineE2ETest, TwoInputMulWorksThroughExecutor) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildBinaryMulGraph()).has_value());
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({2.0f, 3.0f, 4.0f}));
    inputs.emplace("b", Tensor::from_data({5.0f, 6.0f, 7.0f}));
    auto out = engine.infer(inputs);
    ASSERT_TRUE(out.has_value());
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[0], 10.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[2], 28.0f);
}

TEST(EngineE2ETest, ShapeMismatchIsReported) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildMulAddGraph()).has_value());
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    inputs.emplace("b", Tensor::from_data({4.0f, 5.0f}));
    auto out = engine.infer(inputs);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}
