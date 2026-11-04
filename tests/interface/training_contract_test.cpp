#include <gtest/gtest.h>

#include "train/amp/amp.h"
#include "core/plugin/plugin_registry.h"
#include "train/autograd/autograd_engine.h"
#include "train/checkpoint/checkpoint.h"
#include "train/data/dataloader_plugin.h"
#include "train/distributed/distributed.h"
#include "train/loss/loss_plugin.h"
#include "train/optimizer/optimizer_plugin.h"
#include "train/scheduler/lr_scheduler_plugin.h"
#include "train/trainer/trainer.h"

using namespace ie;

struct StubOptimizer : OptimizerPlugin {
    PluginMeta meta() const override { return {"opt", "1.0", "0.1", {}, ""}; }
    std::string optimizer_name() const override { return "stub"; }
    Status step(std::vector<Tensor*>&, std::vector<Tensor*>&,
                const OptimizerConfig&) override {
        return {};
    }
    Status zero_grad(std::vector<Tensor*>&) override { return {}; }
    Status save_state(BufferHandle) override {
        return Error::unsupported("stub");
    }
    Status load_state(BufferHandle) override {
        return Error::unsupported("stub");
    }
};
struct StubScheduler : LRSchedulerPlugin {
    PluginMeta meta() const override { return {"lr", "1.0", "0.1", {}, ""}; }
    std::string scheduler_name() const override { return "stub"; }
    float get_lr(int) override { return 0.1f; }
};
struct StubLoss : LossPlugin {
    PluginMeta meta() const override { return {"loss", "1.0", "0.1", {}, ""}; }
    std::string loss_name() const override { return "stub"; }
    Tensor compute(const Tensor&, const Tensor&) override { return Tensor{}; }
    Tensor backward(const Tensor&) override { return Tensor{}; }
};
struct StubDataloader : DataloaderPlugin {
    PluginMeta meta() const override { return {"data", "1.0", "0.1", {}, ""}; }
    Status load(const std::string&) override { return {}; }
    Result<Batch> next() override { return Batch{}; }
    void shuffle(std::uint64_t) override {}
};

TEST(TrainingPluginRegistriesTest, AllFourRegister) {
    auto o = PluginRegistry<OptimizerPlugin>::instance().register_plugin(
        std::make_unique<StubOptimizer>());
    auto s = PluginRegistry<LRSchedulerPlugin>::instance().register_plugin(
        std::make_unique<StubScheduler>());
    auto l = PluginRegistry<LossPlugin>::instance().register_plugin(
        std::make_unique<StubLoss>());
    auto d = PluginRegistry<DataloaderPlugin>::instance().register_plugin(
        std::make_unique<StubDataloader>());
    EXPECT_TRUE(o.has_value() || o.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(s.has_value() || s.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(l.has_value() || l.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(d.has_value() || d.error().code == Error::Code::AlreadyExists);
}

TEST(TrainingContractsTest, PluginsAndEmptyImplementations) {
    StubOptimizer optimizer;
    std::vector<Tensor*> params, grads;
    EXPECT_TRUE(optimizer.step(params, grads, OptimizerConfig{}).has_value());
    StubScheduler scheduler;
    EXPECT_FLOAT_EQ(scheduler.get_lr(0), 0.1f);
    StubLoss loss;
    EXPECT_TRUE(loss.compute(Tensor{}, Tensor{}).shape.dims.empty());
    StubDataloader dataloader;
    EXPECT_TRUE(dataloader.next().has_value());

    AutogradEngine autograd;
    Tensor tensor;
    EXPECT_EQ(autograd.backward(tensor).error().code, Error::Code::Unsupported);
    EXPECT_EQ(autograd.grad(tensor).error().code, Error::Code::Unsupported);
    EXPECT_FALSE(autograd.is_training());

    Trainer trainer;
    Graph graph;
    DataloaderPlugin& base = dataloader;
    EXPECT_EQ(trainer.train(graph, base, Trainer::Config{}).error().code,
              Error::Code::Unsupported);

    EXPECT_EQ(CheckpointIO::save(Checkpoint{}, "x").error().code,
              Error::Code::Unsupported);
    AmpManager amp;
    EXPECT_EQ(amp.enable(AmpConfig{}).error().code, Error::Code::Unsupported);
    DistributedContext distributed;
    EXPECT_EQ(distributed.init(DistributedConfig{}).error().code,
              Error::Code::Unsupported);
}
