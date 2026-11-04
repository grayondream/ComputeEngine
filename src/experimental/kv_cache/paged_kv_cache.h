#pragma once

#include <cstddef>

#include "core/base/error.h"
#include "core/plugin/kv_cache_plugin.h"

namespace ie {

// Paged KV cache contract (docs/readme.md §11.1). P0.5 freezes the surface;
// block-based storage lands with the experimental KV milestone.
class PagedKVCache final : public KVCachePlugin {
public:
    Status append(int /*layer*/, Tensor /*k*/, Tensor /*v*/) override {
        return Error::unsupported(
            "paged kv cache not implemented in P0.5");
    }

    Result<Tensor> load(int /*layer*/, int /*position*/) override {
        return Error::unsupported(
            "paged kv cache not implemented in P0.5");
    }

    Status clear() override { return {}; }

    std::size_t memory_usage() const override { return 0; }

    PluginMeta meta() const override {
        return PluginMeta{"paged_kv_cache", "0.1.0", "0.1", {"kv_cache"},
                          "P0.5 paged kv-cache placeholder"};
    }
};

}  // namespace ie
