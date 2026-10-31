#pragma once

#include <cstddef>

namespace ie {

// Memory policy, split by model class (docs/readme.md §4.3).
struct MemoryPolicy {
    // Common
    std::size_t tensor_pool_size{0};
    std::size_t activation_pool_size{0};

    // Traditional models
    std::size_t weight_cache_size{0};
    bool enable_weight_prefetch{false};

    // Transformer
    std::size_t kv_cache_block_size{0};
    bool enable_prefix_caching{false};
    std::size_t max_concurrent_requests{0};
};

}  // namespace ie
