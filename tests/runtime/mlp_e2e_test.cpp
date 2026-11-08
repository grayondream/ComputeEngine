#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ir/graph.h"
#include "core/memory/malloc_allocator.h"
#include "core/runtime/engine.h"
#include "core/runtime/executor.h"
#include "data/mlp_reference.h"

using namespace ie;

namespace {

Tensor make_tensor(const Shape& shape, std::span<const float> values) {
    auto alloc = std::make_shared<MallocAllocator>();
    auto tensor = Tensor::allocate(shape, DType::F32, alloc);
    EXPECT_TRUE(tensor.has_value());
    Tensor result = std::move(*tensor);
    std::copy(values.begin(), values.end(), result.data<float>());
    return result;
}

std::vector<float> augment_input() {
    std::vector<float> out(kMlpBatch * (kMlpIn + 1));
    for (int i = 0; i < kMlpBatch; ++i) {
        for (int j = 0; j < kMlpIn; ++j) {
            out[i * (kMlpIn + 1) + j] = kMlpInput[i * kMlpIn + j];
        }
        out[i * (kMlpIn + 1) + kMlpIn] = 1.0f;
    }
    return out;
}

// Builds the [W; b] augmented weight. When carry_constant is true an extra
// trailing 1 column is appended (zero above the bias row) so the layer output
// keeps a homogeneous coordinate for the next bias-folding MatMul. ReLU keeps
// that coordinate exactly 1, so all three layers fold bias without extra ops.
std::vector<float> augment_weight(const float* w, int in, int out,
                                  const float* bias, bool carry_constant) {
    const int columns = out + (carry_constant ? 1 : 0);
    std::vector<float> result((in + 1) * columns, 0.0f);
    for (int i = 0; i < in; ++i) {
        std::copy(w + i * out, w + (i + 1) * out,
                  result.begin() + i * columns);
    }
    for (int j = 0; j < out; ++j) {
        result[in * columns + j] = bias[j];
    }
    if (carry_constant) {
        result[in * columns + out] = 1.0f;
    }
    return result;
}

Graph build_mlp_graph() {
    Graph graph;
    auto value = [&](std::string name, Shape shape, bool input, bool output) {
        Value v;
        v.id = static_cast<ValueId>(graph.values.size());
        v.name = std::move(name);
        v.shape = std::move(shape);
        v.is_input = input;
        v.is_output = output;
        graph.values.push_back(std::move(v));
        return graph.values.back().id;
    };
    auto node = [&](OpType op, std::vector<ValueId> inputs, ValueId output) {
        Node n;
        n.id = static_cast<NodeId>(graph.nodes.size());
        n.op = op;
        n.inputs = std::move(inputs);
        n.outputs = {output};
        graph.nodes.push_back(std::move(n));
    };

    // Hidden activations carry one extra constant-1 column so the next layer
    // can fold its bias through the same homogeneous coordinate.
    constexpr int h1_width = kMlpH1 + 1;
    constexpr int h2_width = kMlpH2 + 1;
    const ValueId x = value("x", Shape{{kMlpBatch, kMlpIn + 1}}, true, false);
    const ValueId w1 = value("w1", Shape{{kMlpIn + 1, h1_width}}, true, false);
    const ValueId w2 = value("w2", Shape{{kMlpH1 + 1, h2_width}}, true, false);
    const ValueId w3 = value("w3", Shape{{kMlpH2 + 1, kMlpOut}}, true, false);
    const ValueId h1 = value("h1", Shape{{kMlpBatch, h1_width}}, false, false);
    const ValueId h1r = value("h1r", Shape{{kMlpBatch, h1_width}}, false, false);
    const ValueId h2 = value("h2", Shape{{kMlpBatch, h2_width}}, false, false);
    const ValueId h2r = value("h2r", Shape{{kMlpBatch, h2_width}}, false, false);
    const ValueId logits = value("logits", Shape{{kMlpBatch, kMlpOut}}, false,
                                 false);
    const ValueId probs = value("probs", Shape{{kMlpBatch, kMlpOut}}, false,
                                true);

    node(OpType::MatMul, {x, w1}, h1);
    node(OpType::Relu, {h1}, h1r);
    node(OpType::MatMul, {h1r, w2}, h2);
    node(OpType::Relu, {h2}, h2r);
    node(OpType::MatMul, {h2r, w3}, logits);
    node(OpType::Softmax, {logits}, probs);
    return graph;
}

std::unordered_map<std::string, Tensor> make_inputs() {
    const std::vector<float> x = augment_input();
    const std::vector<float> w1 =
        augment_weight(kMlpW1, kMlpIn, kMlpH1, kMlpB1, true);
    const std::vector<float> w2 =
        augment_weight(kMlpW2, kMlpH1, kMlpH2, kMlpB2, true);
    const std::vector<float> w3 =
        augment_weight(kMlpW3, kMlpH2, kMlpOut, kMlpB3, false);
    return {
        {"x", make_tensor(Shape{{kMlpBatch, kMlpIn + 1}}, x)},
        {"w1", make_tensor(Shape{{kMlpIn + 1, kMlpH1 + 1}}, w1)},
        {"w2", make_tensor(Shape{{kMlpH1 + 1, kMlpH2 + 1}}, w2)},
        {"w3", make_tensor(Shape{{kMlpH2 + 1, kMlpOut}}, w3)},
    };
}

}  // namespace

TEST(MlpE2ETest, MatchesPyTorchReferenceThroughEngine) {
    Engine engine(EngineConfig{});
    Graph graph = build_mlp_graph();
    ASSERT_TRUE(engine.load(graph).has_value());
    auto outputs = engine.infer(make_inputs());
    ASSERT_TRUE(outputs.has_value()) << outputs.error().message;
    ASSERT_EQ(outputs->size(), 1u);
    const Tensor& y = outputs->front();
    ASSERT_EQ(y.shape.dims, (std::vector<int64_t>{kMlpBatch, kMlpOut}));
    for (int i = 0; i < kMlpBatch * kMlpOut; ++i) {
        EXPECT_NEAR(y.data<float>()[i], kMlpExpected[i], 1e-5f);
    }
}

TEST(MlpE2ETest, ScalarPathMatchesGolden) {
    auto alloc = std::make_shared<MallocAllocator>();
    Executor executor;
    auto outputs = executor.run(build_mlp_graph(), make_inputs(), alloc);
    ASSERT_TRUE(outputs.has_value()) << outputs.error().message;
    ASSERT_EQ(outputs->size(), 1u);
    for (int i = 0; i < kMlpBatch * kMlpOut; ++i) {
        EXPECT_NEAR(outputs->front().data<float>()[i], kMlpExpected[i], 1e-5f);
    }
}
