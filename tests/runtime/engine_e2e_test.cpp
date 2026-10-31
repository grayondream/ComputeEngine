#include <gtest/gtest.h>

#include "core/ir/graph_builder.h"
#include "core/runtime/engine.h"

using namespace ie;

static Graph BuildMulAddGraph() {
    GraphBuilder b;
    return std::move(b).add(2).mul(3.0).build();
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
