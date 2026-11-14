#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/plugin/plugin_registry.h"
#include "core/plugin/ring_kv_cache.h"
#include "engine/traditional/traditional_engine.h"
#include "experimental/kv_cache/paged_kv_cache.h"
#include "experimental/speculative/speculative_decoder.h"
#include "llm/model_loader/model_loader.h"
#include "llm/sampler/sampler.h"
#include "llm/streaming/streaming.h"
#include "llm/tokenizer/tokenizer.h"
#include "llm/transformer/transformer_model.h"

using namespace ie;

namespace {

struct StubTokenizer : Tokenizer {
    Result<std::vector<int>> encode(std::string_view) override {
        return std::vector<int>{1, 2};
    }
    Result<std::string> decode(std::span<const int>) override {
        return std::string("x");
    }
};

struct StubSampler : Sampler {
    int sample(std::span<const float>, const SamplingParams&) override {
        return 0;
    }
};

struct StubModelLoader : ModelLoader {
    Result<Graph> load(const ModelConfig&) override { return Graph{}; }
};

struct StubTransformerModel : TransformerModel {
    Status load(const ModelConfig&) override { return {}; }
    Status forward(const BatchPlan&, ExecutionContext&) override { return {}; }
};

struct StubLLMEngine : LLMEngine {
    Status load(const ModelConfig&) override { return {}; }
    Status generate(std::string_view, const GenerationParams&,
                    std::function<void(std::string_view)> on_token) override {
        on_token("x");
        return {};
    }
};

struct StubStream : TokenStream {
    int tokens = 0;
    bool ended = false;
    void on_token(std::string_view) override { ++tokens; }
    void on_end() override { ended = true; }
};

struct StubTokenizerPlugin : TokenizerPlugin {
    PluginMeta meta() const override { return {"tokenizer", "1.0", "0.1", {}, ""}; }
    std::unique_ptr<Tokenizer> create(const TokenizerConfig&) override {
        return nullptr;
    }
};

struct StubSamplerPlugin : SamplerPlugin {
    PluginMeta meta() const override { return {"sampler", "1.0", "0.1", {}, ""}; }
    std::unique_ptr<Sampler> create() override { return nullptr; }
};

struct StubModelLoaderPlugin : ModelLoaderPlugin {
    PluginMeta meta() const override { return {"loader", "1.0", "0.1", {}, ""}; }
    std::unique_ptr<ModelLoader> create() override { return nullptr; }
};

}  // namespace

TEST(LlmPluginRegistriesTest, TokenizerSamplerLoaderKvCache) {
    auto t = PluginRegistry<TokenizerPlugin>::instance().register_plugin(
        std::make_unique<StubTokenizerPlugin>());
    auto s = PluginRegistry<SamplerPlugin>::instance().register_plugin(
        std::make_unique<StubSamplerPlugin>());
    auto m = PluginRegistry<ModelLoaderPlugin>::instance().register_plugin(
        std::make_unique<StubModelLoaderPlugin>());
    auto k = PluginRegistry<KVCachePlugin>::instance().register_plugin(
        std::make_unique<RingKVCache>());
    EXPECT_TRUE(t.has_value() || t.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(s.has_value() || s.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(m.has_value() || m.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(k.has_value() || k.error().code == Error::Code::AlreadyExists);
}

TEST(LlmContractsTest, PureInterfacesHaveUsableStubs) {
    StubTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.encode("a").has_value());
    EXPECT_EQ(tokenizer.encode("a")->size(), 2u);
    EXPECT_EQ(tokenizer.decode(std::span<const int>{}).value(), "x");

    StubSampler sampler;
    float logits[] = {0.1f, 0.9f};
    EXPECT_EQ(sampler.sample(logits, SamplingParams{}), 0);

    StubModelLoader loader;
    EXPECT_TRUE(loader.load(ModelConfig{}).has_value());

    StubTransformerModel model;
    EXPECT_TRUE(model.load(ModelConfig{}).has_value());

    StubStream stream;
    StubLLMEngine engine;
    EXPECT_TRUE(engine.load(ModelConfig{}).has_value());
    EXPECT_TRUE(engine.generate("p", GenerationParams{}, [&](std::string_view t) {
        stream.on_token(t);
    }).has_value());
    stream.on_end();
    EXPECT_EQ(stream.tokens, 1);
    EXPECT_TRUE(stream.ended);
}

TEST(ExperimentalLlmTest, DefaultsAreUnsupported) {
    PagedKVCache cache;
    EXPECT_EQ(cache.append(0, Tensor{}, Tensor{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(cache.load(0, 0).error().code, Error::Code::Unsupported);
    EXPECT_TRUE(cache.clear().has_value());
    EXPECT_EQ(cache.memory_usage(), 0u);
    EXPECT_EQ(cache.meta().name, "paged_kv_cache");
    EXPECT_EQ(cache.meta().version, "0.1.0");
    EXPECT_EQ(cache.meta().api_version, "0.1");

    SpeculativeDecoder decoder;
    float logits[] = {1.0f};
    EXPECT_EQ(decoder.decode(logits, {}).error().code,
              Error::Code::Unsupported);
}

TEST(TraditionalEngineTest, UnloadedUseIsRejected) {
    TraditionalEngine engine;
    EXPECT_EQ(engine.load_model("m", ModelFormat::ONNX).error().code,
              Error::Code::InvalidArgument);
    EXPECT_EQ(engine.infer({}).error().code, Error::Code::InvalidArgument);
    // An empty batch is valid and produces no results.
    auto empty = engine.infer_batch({});
    ASSERT_TRUE(empty.has_value());
    EXPECT_TRUE(empty->empty());
}
