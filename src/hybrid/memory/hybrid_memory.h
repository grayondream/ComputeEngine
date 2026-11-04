#pragma once

#include <cstddef>

#include "core/base/error.h"
#include "core/base/types.h"
#include "core/memory/allocator.h"
#include "hybrid/hybrid_config.h"

namespace ie {

// Cross-device memory management for hybrid execution (docs/readme.md §7.5).
// configure() accepts the configuration; allocation is not implemented in P0.5.
class HybridMemoryManager {
public:
    Status configure(const HybridConfig&) { return {}; }

    Result<BufferHandle> allocate(std::size_t, Device) {
        return Error::unsupported("hybrid memory allocation not implemented in P0.5");
    }

    Status release(BufferHandle) { return {}; }
};

}  // namespace ie
