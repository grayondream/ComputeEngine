#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "core/plugin/kv_cache_types.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

class KVCachePlugin {
public:
    virtual ~KVCachePlugin() = default;

    virtual Status append(int layer, Tensor k, Tensor v) = 0;
    virtual Result<Tensor> load(int layer, int position) = 0;
    virtual Status clear() = 0;
    virtual std::size_t memory_usage() const = 0;

    virtual BlockTable block_table() const { return {}; }
    virtual Status evict_blocks(std::span<const BlockId> /*blocks*/) {
        return {};
    }
    virtual Status compress(int /*layer*/, QuantFormat /*format*/) { return {}; }
    virtual Result<std::vector<BlockId>> lookup_prefix(BlockHash /*hash*/) {
        return std::vector<BlockId>{};
    }
    virtual void register_prefix(BlockHash /*hash*/,
                                 std::span<const BlockId> /*blocks*/) {}
    virtual Status offload(int /*layer*/, int /*start*/, int /*end*/) {
        return {};
    }
    virtual Status prefetch(int /*layer*/, int /*start*/, int /*end*/) {
        return {};
    }
    virtual Result<TensorHash> hash_state() const { return TensorHash{0}; }

    virtual PluginMeta meta() const = 0;
};

}  // namespace ie
