#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

// Type-level contracts frozen by P0.5.
static_assert(std::is_same_v<decltype(OptimizerConfig::lr), float>);
static_assert(std::is_same_v<decltype(OptimizerConfig::weight_decay), float>);
static_assert(std::is_same_v<decltype(OptimizerConfig::beta1), float>);
static_assert(std::is_same_v<decltype(OptimizerConfig::beta2), float>);
static_assert(std::is_same_v<decltype(OptimizerConfig::eps), float>);
static_assert(std::is_same_v<decltype(OptimizerConfig::momentum), float>);

static_assert(std::is_same_v<decltype(Trainer::Config::epochs), int>);
static_assert(std::is_same_v<decltype(Trainer::Config::batch_size), int>);
static_assert(std::is_same_v<decltype(Trainer::Config::lr), float>);
static_assert(std::is_same_v<decltype(Trainer::Config::optimizer), std::string>);
static_assert(std::is_same_v<decltype(Trainer::Config::scheduler), std::string>);
static_assert(std::is_same_v<decltype(Trainer::Config::loss), std::string>);
static_assert(std::is_same_v<decltype(Trainer::Config::use_amp), bool>);
static_assert(std::is_same_v<decltype(Trainer::Config::grad_accum_steps), int>);
static_assert(std::is_same_v<decltype(Trainer::Config::checkpoint_interval), int>);
static_assert(std::is_same_v<decltype(Trainer::Config::checkpoint_dir),
                             std::string>);

static_assert(std::is_same_v<decltype(AmpConfig::enabled), bool>);
static_assert(std::is_same_v<decltype(AmpConfig::compute_dtype), DType>);
static_assert(std::is_same_v<decltype(AmpConfig::loss_scale), float>);

static_assert(std::is_same_v<decltype(DistributedConfig::world_size), int>);
static_assert(std::is_same_v<decltype(DistributedConfig::rank), int>);
static_assert(std::is_same_v<decltype(DistributedConfig::backend),
                             std::string>);

static_assert(std::is_same_v<decltype(Metrics::values),
                             std::unordered_map<std::string, double>>);

static_assert(std::is_same_v<decltype(Checkpoint::model_weights),
                             std::unordered_map<std::string, Tensor>>);
static_assert(
    std::is_same_v<decltype(Checkpoint::optimizer_state),
                   std::optional<std::unordered_map<std::string, Tensor>>>);
static_assert(
    std::is_same_v<decltype(Checkpoint::scheduler_state),
                   std::optional<std::unordered_map<std::string, Tensor>>>);
static_assert(
    std::is_same_v<decltype(Checkpoint::rng_state),
                   std::optional<std::unordered_map<std::string, BufferHandle>>>);
static_assert(std::is_same_v<decltype(Checkpoint::global_step),
                             std::optional<int>>);
static_assert(std::is_same_v<decltype(Checkpoint::epoch), std::optional<int>>);
static_assert(std::is_same_v<decltype(Checkpoint::graph),
                             std::optional<Graph>>);

namespace {

struct StubOp : Op {
    Status forward(const std::vector<Tensor*>&, std::vector<Tensor*>&,
                   ExecutionContext&) override {
        return Error::unsupported("stub");
    }
};

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

}  // namespace

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

TEST(TrainingDefaultsTest, OptimizerConfigDefaults) {
    OptimizerConfig config;
    EXPECT_FLOAT_EQ(config.lr, 1e-3f);
    EXPECT_FLOAT_EQ(config.weight_decay, 0.0f);
    EXPECT_FLOAT_EQ(config.beta1, 0.9f);
    EXPECT_FLOAT_EQ(config.beta2, 0.999f);
    EXPECT_FLOAT_EQ(config.eps, 1e-8f);
    EXPECT_FLOAT_EQ(config.momentum, 0.0f);
}

TEST(TrainingDefaultsTest, TrainerConfigDefaults) {
    Trainer::Config config;
    EXPECT_EQ(config.epochs, 10);
    EXPECT_EQ(config.batch_size, 32);
    EXPECT_FLOAT_EQ(config.lr, 1e-4f);
    EXPECT_EQ(config.optimizer, "adamw");
    EXPECT_EQ(config.scheduler, "cosine");
    EXPECT_EQ(config.loss, "cross_entropy");
    EXPECT_FALSE(config.use_amp);
    EXPECT_EQ(config.grad_accum_steps, 1);
    EXPECT_EQ(config.checkpoint_interval, 1000);
    EXPECT_TRUE(config.checkpoint_dir.empty());
}

TEST(TrainingDefaultsTest, AmpDefaultsAndEnableIsUnsupported) {
    AmpConfig config;
    EXPECT_FALSE(config.enabled);
    EXPECT_EQ(config.compute_dtype, DType::F16);
    EXPECT_FLOAT_EQ(config.loss_scale, 1.0f);

    AmpManager amp;
    EXPECT_FALSE(amp.is_enabled());
    EXPECT_EQ(amp.enable(config).error().code, Error::Code::Unsupported);
    EXPECT_FALSE(amp.is_enabled());
}

TEST(TrainingDefaultsTest, DistributedDefaultsAndQueries) {
    DistributedConfig config;
    EXPECT_EQ(config.world_size, 1);
    EXPECT_EQ(config.rank, 0);
    EXPECT_TRUE(config.backend.empty());

    DistributedContext context;
    EXPECT_EQ(context.rank(), 0);
    EXPECT_EQ(context.world_size(), 1);
    EXPECT_EQ(context.barrier().error().code, Error::Code::Unsupported);
    Tensor tensor;
    EXPECT_EQ(context.all_reduce(tensor, "sum").error().code,
              Error::Code::Unsupported);
}

TEST(TrainingDefaultsTest, AutogradTrainingStateAndNoOpRecording) {
    AutogradEngine engine;
    EXPECT_FALSE(engine.is_training());

    engine.set_training(true);
    EXPECT_TRUE(engine.is_training());

    engine.zero_grad();
    StubOp op;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    engine.record(op, inputs, outputs);
    EXPECT_TRUE(engine.is_training());

    engine.set_training(false);
    EXPECT_FALSE(engine.is_training());
}

TEST(TrainingDefaultsTest, CheckpointStartsEmpty) {
    Checkpoint checkpoint;
    EXPECT_TRUE(checkpoint.model_weights.empty());
    EXPECT_FALSE(checkpoint.optimizer_state.has_value());
    EXPECT_FALSE(checkpoint.scheduler_state.has_value());
    EXPECT_FALSE(checkpoint.rng_state.has_value());
    EXPECT_FALSE(checkpoint.global_step.has_value());
    EXPECT_FALSE(checkpoint.epoch.has_value());
    EXPECT_FALSE(checkpoint.graph.has_value());
}
