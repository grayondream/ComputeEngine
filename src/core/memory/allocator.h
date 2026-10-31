#pragma once

#include <cstddef>

#include "core/base/error.h"
#include "core/memory/memory_kind.h"

namespace ie {

using BufferHandle = void*;

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual Result<BufferHandle> alloc(std::size_t bytes, MemoryKind kind) = 0;
    virtual void free(BufferHandle handle) = 0;
    virtual std::size_t available(MemoryKind kind) const = 0;

    virtual Status free_gradients() { return {}; }
    virtual Status free_activations() { return {}; }
};

}  // namespace ie
