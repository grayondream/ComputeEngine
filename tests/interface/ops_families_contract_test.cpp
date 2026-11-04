#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "experimental/attention/flash_attention.h"
#include "experimental/moe/moe_dispatch.h"
#include "core/ir/op_plugin.h"
#include "core/plugin/plugin_registry.h"
#include "ops/control/control_ops.h"
#include "ops/traditional/traditional_ops.h"
#include "ops/transformer/transformer_ops.h"

using namespace ie;

TEST(OpsFamiliesTest, EveryStubReturnsUnsupported) {
    std::vector<std::unique_ptr<Op>> ops;
    ops.push_back(std::make_unique<Conv2DOp>());
    ops.push_back(std::make_unique<LSTMOp>());
    ops.push_back(std::make_unique<GBDTLeafOp>());
    ops.push_back(std::make_unique<AttentionOp>());
    ops.push_back(std::make_unique<MoEExpertOp>());
    ops.push_back(std::make_unique<IfOp>());
    ops.push_back(std::make_unique<FlashAttentionOp>());
    ops.push_back(std::make_unique<MoEDispatchOp>());

    std::vector<Tensor*> in;
    std::vector<Tensor*> out;
    ExecutionContext context;
    for (const auto& op : ops) {
        EXPECT_EQ(op->forward(in, out, context).error().code,
                  Error::Code::Unsupported);
    }
}

TEST(ControlOpsTest, ControlFlowFlagIsTrue) {
    IfOp if_op;
    LoopOp loop_op;
    EXPECT_TRUE(if_op.is_control_flow());
    EXPECT_TRUE(loop_op.is_control_flow());
    ExecutionContext context;
    EXPECT_EQ(if_op.execute_subgraph(context, Subgraph{}).error().code,
              Error::Code::Unsupported);
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
