#pragma once

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "core/memory/allocator.h"

namespace ie {

// P0 allocator: a thin malloc wrapper that tracks outstanding blocks.
class MallocAllocator final : public Allocator {
public:
    Result<BufferHandle> alloc(std::size_t bytes, MemoryKind kind) override {
        if (bytes == 0) {
            return Error::invalid_argument("alloc size is zero");
        }
        void* raw = std::malloc(bytes);
        if (raw == nullptr) {
            return Error::out_of_memory("malloc failed");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        blocks_.emplace(raw, bytes);
        bytes_ += bytes;
        (void)kind;
        return raw;
    }

    void free(BufferHandle handle) override {
        if (handle == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = blocks_.find(handle);
        if (it == blocks_.end()) {
            return;
        }
        bytes_ -= it->second;
        blocks_.erase(it);
        std::free(handle);
    }

    std::size_t available(MemoryKind /*kind*/) const override {
        return static_cast<std::size_t>(-1);
    }

    std::size_t outstanding_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocks_.size();
    }

    std::size_t outstanding_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bytes_;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, std::size_t> blocks_;
    std::size_t bytes_{0};
};

}  // namespace ie
