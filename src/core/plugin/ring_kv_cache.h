#pragma once

#include "core/plugin/kv_cache_plugin.h"

namespace ie {

// P0 placeholder: the ring cache contract exists, storage lands in P6.
class RingKVCache final : public KVCachePlugin {
public:
    Status append(int /*layer*/, Tensor /*k*/, Tensor /*v*/) override {
        return Error::unsupported("kv cache not implemented in P0");
    }

    Result<Tensor> load(int /*layer*/, int /*position*/) override {
        return Error::unsupported("kv cache not implemented in P0");
    }

    Status clear() override { return {}; }

    std::size_t memory_usage() const override { return 0; }

    PluginMeta meta() const override {
        return PluginMeta{"ring_kv_cache", "0.1.0", "0.1", {"kv_cache"},
                          "P0 ring kv-cache placeholder"};
    }
};

}  // namespace ie
