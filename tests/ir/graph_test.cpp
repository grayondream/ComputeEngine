#include <gtest/gtest.h>

#include "core/ir/graph_builder.h"

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
