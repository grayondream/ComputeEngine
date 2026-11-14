#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "experimental/attention/flash_attention.h"
#include "experimental/moe/moe_dispatch.h"
#include "core/ir/op_plugin.h"
#include "core/plugin/plugin_registry.h"
#include "ops/control/control_ops.h"
#include "ops/traditional/traditional_ops.h"
#include "ops/transformer/transformer_ops.h"

using namespace ie;

namespace {

struct StubCase {
    const char* name;
    std::function<std::unique_ptr<Op>()> make;
};

const std::vector<StubCase>& AllStubCases() {
    static const std::vector<StubCase> cases = {
        // traditional
        {"ConvTranspose2D",
         [] { return std::make_unique<ConvTranspose2DOp>(); }},
        {"LayerNorm", [] { return std::make_unique<LayerNormOp>(); }},
        {"GroupNorm", [] { return std::make_unique<GroupNormOp>(); }},
        {"LSTM", [] { return std::make_unique<LSTMOp>(); }},
        {"GRU", [] { return std::make_unique<GRUOp>(); }},
        {"RNN", [] { return std::make_unique<RNNOp>(); }},
        {"GBDTSplit", [] { return std::make_unique<GBDTSplitOp>(); }},
        {"GBDTLeaf", [] { return std::make_unique<GBDTLeafOp>(); }},
        // transformer
        {"Attention", [] { return std::make_unique<AttentionOp>(); }},
        {"MultiHeadAttention",
         [] { return std::make_unique<MultiHeadAttentionOp>(); }},
        {"RoPE", [] { return std::make_unique<RoPEOp>(); }},
        {"RMSNorm", [] { return std::make_unique<RMSNormOp>(); }},
        {"Embedding", [] { return std::make_unique<EmbeddingOp>(); }},
        {"PositionalEncoding",
         [] { return std::make_unique<PositionalEncodingOp>(); }},
        {"MoERouter", [] { return std::make_unique<MoERouterOp>(); }},
        {"MoEExpert", [] { return std::make_unique<MoEExpertOp>(); }},
        // control flow
        {"If", [] { return std::make_unique<IfOp>(); }},
        {"Loop", [] { return std::make_unique<LoopOp>(); }},
        {"While", [] { return std::make_unique<WhileOp>(); }},
        {"Subgraph", [] { return std::make_unique<SubgraphOp>(); }},
        // experimental
        {"FlashAttention",
         [] { return std::make_unique<FlashAttentionOp>(); }},
        {"MoEDispatch", [] { return std::make_unique<MoEDispatchOp>(); }},
    };
    return cases;
}

struct ControlCase {
    const char* name;
    std::function<std::unique_ptr<Op>()> make;
};

const std::vector<ControlCase>& AllControlCases() {
    static const std::vector<ControlCase> cases = {
        {"If", [] { return std::make_unique<IfOp>(); }},
        {"Loop", [] { return std::make_unique<LoopOp>(); }},
        {"While", [] { return std::make_unique<WhileOp>(); }},
        {"Subgraph", [] { return std::make_unique<SubgraphOp>(); }},
    };
    return cases;
}

}  // namespace

TEST(OpsFamiliesTest, AllStubsReturnExactUnsupportedMessage) {
    std::vector<Tensor*> in;
    std::vector<Tensor*> out;
    ExecutionContext context;

    ASSERT_EQ(AllStubCases().size(), 22u);
    for (const StubCase& test_case : AllStubCases()) {
        SCOPED_TRACE(test_case.name);
        auto op = test_case.make();
        auto status = op->forward(in, out, context);
        EXPECT_EQ(status.error().code, Error::Code::Unsupported);
        EXPECT_EQ(status.error().message,
                  std::string(test_case.name) + " not implemented in P0.5");
    }
}

TEST(ControlOpsTest, ControlFlowFlagIsTrueAndSubgraphIsUnsupported) {
    ExecutionContext context;
    for (const ControlCase& test_case : AllControlCases()) {
        SCOPED_TRACE(test_case.name);
        auto op = test_case.make();
        EXPECT_TRUE(op->is_control_flow());
        EXPECT_EQ(op->execute_subgraph(context, Subgraph{}).error().code,
                  Error::Code::Unsupported);
    }
}

struct StubOpPlugin : OpPlugin {
    PluginMeta meta() const override { return {"op", "1.0", "0.1", {}, ""}; }
    Result<std::unique_ptr<Op>> create(const Attrs&) override {
        return Error::unsupported("stub");
    }
};

TEST(OpPluginRegistryTest, RegistersCompatiblePlugin) {
    auto& registry = PluginRegistry<OpPlugin>::instance();
    auto status = registry.register_plugin(std::make_unique<StubOpPlugin>());
    EXPECT_TRUE(status.has_value() ||
                status.error().code == Error::Code::AlreadyExists);
}
