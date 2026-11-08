#pragma once

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "core/memory/allocator.h"

namespace ie {

// P1 host allocator: every returned block is aligned to kAlignment bytes.
class CpuAllocator final : public Allocator {
public:
    static constexpr std::size_t kAlignment = 64;

    Result<BufferHandle> alloc(std::size_t bytes, MemoryKind kind) override {
        if (bytes == 0) {
            return Error::invalid_argument("alloc size is zero");
        }
        if (bytes > static_cast<std::size_t>(-1) - (kAlignment - 1)) {
            return Error::out_of_memory(
                "alloc size overflows alignment rounding");
        }
        const std::size_t aligned = round_up(bytes);

        void* raw = nullptr;
#if defined(_WIN32)
        raw = _aligned_malloc(aligned, kAlignment);
#else
        if (posix_memalign(&raw, kAlignment, aligned) != 0) {
            raw = nullptr;
        }
#endif
        if (raw == nullptr) {
            return Error::out_of_memory("aligned allocation failed");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        blocks_.emplace(raw, aligned);
        bytes_ += aligned;
        (void)kind;
        return raw;
    }

    void free(BufferHandle handle) override {
        if (handle == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = blocks_.find(handle);
            if (it == blocks_.end()) {
                return;
            }
            bytes_ -= it->second;
            blocks_.erase(it);
        }
#if defined(_WIN32)
        _aligned_free(handle);
#else
        std::free(handle);
#endif
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
    static constexpr std::size_t round_up(std::size_t bytes) {
        return (bytes + kAlignment - 1) / kAlignment * kAlignment;
    }

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::size_t> blocks_;
    std::size_t bytes_{0};
};

}  // namespace ie
