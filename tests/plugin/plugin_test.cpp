#include <gtest/gtest.h>

#include "core/base/rng.h"
#include "core/plugin/plugin_registry.h"
#include "core/plugin/ring_kv_cache.h"

using namespace ie;

struct FakePlugin {
    PluginMeta m{"fake", "1.0", "0.1", {"test"}, "fake plugin"};
    PluginMeta meta() const { return m; }
};

TEST(PluginRegistryTest, RegisterGetList) {
    auto& reg = PluginRegistry<FakePlugin>::instance();
    ASSERT_TRUE(reg.register_plugin(std::make_unique<FakePlugin>()).has_value());
    auto p = reg.get("fake");
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ((*p)->meta().version, "1.0");
    EXPECT_EQ(reg.list_by_tag("test").size(), 1u);
}

TEST(PluginRegistryTest, DuplicateRejected) {
    auto& reg = PluginRegistry<FakePlugin>::instance();
    reg.register_plugin(std::make_unique<FakePlugin>());
    auto st = reg.register_plugin(std::make_unique<FakePlugin>());
    EXPECT_EQ(st.error().code, Error::Code::AlreadyExists);
}

TEST(RingKVCacheTest, AppendUnsupportedInP0) {
    RingKVCache kv;
    EXPECT_EQ(kv.append(0, Tensor{}, Tensor{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(kv.memory_usage(), 0u);
}

TEST(SamplingRNGTest, SameSeedSameSequence) {
    SamplingRNG a;
    SamplingRNG b;
    a.set_seed(42);
    b.set_seed(42);
    EXPECT_EQ(a.next_u64(), b.next_u64());
    a.set_seed(7);
    b.set_seed(7);
    EXPECT_FLOAT_EQ(a.next_float(), b.next_float());
}
