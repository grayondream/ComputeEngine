#include <gtest/gtest.h>

#include "core/backend/engine_config.h"
#include "core/config/feature_flags.h"
#include "core/config/model_format.h"
#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/memory/memory_kind.h"
#include "core/memory/policy.h"
#include "core/plugin/plugin_api.h"
#include "core/plugin/plugin_registry.h"

using namespace ie;

struct VersionedPlugin {
    PluginMeta m{"versioned", "1.0", "0.1", {}, ""};
    PluginMeta meta() const { return m; }
};
struct IncompatiblePlugin {
    PluginMeta m{"incompatible", "1.0", "9.9", {}, ""};
    PluginMeta meta() const { return m; }
};

TEST(FeatureFlagsTest, DefaultsMatchDesign) {
    FeatureFlags f;
    EXPECT_FALSE(f.enable_experimental);
    EXPECT_FALSE(f.enable_fp8);
    EXPECT_FALSE(f.enable_speculative);
    EXPECT_FALSE(f.enable_paged_attention);
    EXPECT_FALSE(f.enable_moe);
    EXPECT_FALSE(f.enable_training);
    EXPECT_FALSE(f.enable_amp);
    EXPECT_FALSE(f.enable_distributed);
    EXPECT_FALSE(f.enable_validation);
    EXPECT_FALSE(f.enable_deterministic);
    EXPECT_FALSE(f.enable_reference_trace);
    EXPECT_TRUE(f.enable_capability_driven_selection);
    EXPECT_TRUE(f.enable_runtime_autotuning);
    EXPECT_TRUE(f.enable_conv_winograd);
    EXPECT_FALSE(f.enable_conv_fft);
    EXPECT_TRUE(f.enable_int8_ptq);
    EXPECT_TRUE(f.enable_flash_attention);
    EXPECT_TRUE(f.enable_kv_cache_quant);
    EXPECT_FALSE(global_feature_flags().enable_training);
}

TEST(PluginApiTest, CompatibilityRules) {
    EXPECT_TRUE(is_api_compatible("0.1"));
    EXPECT_TRUE(is_api_compatible("0.0"));
    EXPECT_FALSE(is_api_compatible("0.2"));
    EXPECT_FALSE(is_api_compatible("1.0"));
    EXPECT_FALSE(is_api_compatible("garbage"));
}

TEST(PluginRegistryApiTest, RejectsIncompatiblePlugin) {
    auto& reg = PluginRegistry<IncompatiblePlugin>::instance();
    auto status = reg.register_plugin(std::make_unique<IncompatiblePlugin>());
    ASSERT_FALSE(status.has_value());
    EXPECT_EQ(status.error().code, Error::Code::InvalidArgument);
}

TEST(PluginRegistryApiTest, AcceptsCompatiblePlugin) {
    auto& reg = PluginRegistry<VersionedPlugin>::instance();
    EXPECT_TRUE(reg.register_plugin(std::make_unique<VersionedPlugin>()).has_value());
}

TEST(EngineConfigFeatureTest, CarriesFlags) {
    EngineConfig cfg;
    EXPECT_FALSE(cfg.features.enable_training);
    EXPECT_TRUE(cfg.features.enable_flash_attention);
}

// FeatureFlags is the authoritative gate; the legacy bools are per-engine
// mirrors and must start in sync with it.
TEST(EngineConfigFeatureTest, DefaultMirrorsMatchFeatureFlags) {
    EngineConfig cfg;
    EXPECT_EQ(cfg.enable_training, cfg.features.enable_training);
    EXPECT_EQ(cfg.enable_validation, cfg.features.enable_validation);
    EXPECT_EQ(cfg.enable_deterministic, cfg.features.enable_deterministic);
    EXPECT_EQ(cfg.hardware_adapt.enable_runtime_autotuning,
              cfg.features.enable_runtime_autotuning);
    EXPECT_EQ(cfg.hardware_adapt.enable_capability_driven_selection,
              cfg.features.enable_capability_driven_selection);
    EXPECT_EQ(cfg.validation.enabled, cfg.features.enable_validation);
}

TEST(ModelFormatTest, HasAllDesignFormats) {
    EXPECT_NE(ModelFormat::ONNX, ModelFormat::GGUF);
    EXPECT_NE(ModelFormat::SAFETENSORS, ModelFormat::CUSTOM);
}

TEST(DualModelFieldsTest, EnumsAndPolicyExist) {
    EXPECT_NE(OpType::Conv2D, OpType::Attention);
    EXPECT_NE(OpType::GBDT_Split, OpType::RoPE);
    EXPECT_NE(MemoryKind::WEIGHT_CONST, MemoryKind::KV_CACHE);
    EXPECT_NE(ModelKind::TRADITIONAL, ModelKind::TRANSFORMER);
    MemoryPolicy policy;
    EXPECT_EQ(policy.kv_cache_block_size, 0u);
    EXPECT_FALSE(policy.enable_weight_prefetch);
}
