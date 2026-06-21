#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/memory/allocator.h"
#include <cstddef>
#include <vector>
#include <set>
#include <mutex>

namespace compute_engine {

struct PoolStats {
    size_t total_allocated = 0;
    size_t peak_allocated = 0;
    size_t current_allocated = 0;
    size_t num_allocations = 0;
    size_t num_deallocations = 0;
};

class MemoryPool {
public:
    explicit MemoryPool(Allocator* upstream = nullptr,
                        size_t small_pool_max = 1 << 20,
                        size_t large_pool_min = 256 * 1024,
                        size_t alignment = 256);
    ~MemoryPool() = default;

    Status Allocate(size_t size, void** ptr);
    Status Deallocate(void* ptr, size_t size);
    void Trim();
    PoolStats GetStats() const;

    static constexpr size_t kDefaultSmallPoolMax = 1 << 20;
    static constexpr size_t kDefaultLargePoolMin = 256 * 1024;
    static constexpr size_t kDefaultAlignment = 256;

private:
    class SmallPool {
    public:
        explicit SmallPool(Allocator* upstream, size_t alignment);
        Status Allocate(size_t size, void** ptr);
        Status Deallocate(void* ptr, size_t size);
        void Trim();
    private:
        static constexpr int kNumBuckets = 14;
        std::vector<void*> free_list_[kNumBuckets];
        Allocator* upstream_;
        size_t alignment_;
        int BucketIndex(size_t size) const;
        size_t BucketSize(int index) const;
    };

    class LargePool {
    public:
        explicit LargePool(Allocator* upstream);
        Status Allocate(size_t size, void** ptr);
        Status Deallocate(void* ptr, size_t size);
        void Trim();
    private:
        struct Block { void* ptr; size_t size; };
        std::set<Block> free_blocks_;
        Allocator* upstream_;
    };

    SmallPool small_;
    LargePool large_;
    Allocator* upstream_;
    size_t small_pool_max_;
    mutable std::mutex mutex_;
    PoolStats stats_;
};

} // namespace compute_engine
