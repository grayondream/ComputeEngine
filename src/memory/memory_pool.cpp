#include "compute_engine/memory/memory_pool.h"
#include "compute_engine/platform/logger.h"
#include <algorithm>
#include <cmath>

namespace compute_engine {

MemoryPool::MemoryPool(Allocator* upstream, size_t small_pool_max,
                       size_t large_pool_min, size_t alignment)
    : small_(upstream, alignment), large_(upstream),
      upstream_(upstream), small_pool_max_(small_pool_max) {
    LOGI("MemoryPool created (small_max={}, large_min={})", small_pool_max_, large_pool_min);
}

Status MemoryPool::Allocate(size_t size, void** ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (size < small_pool_max_) {
        auto status = small_.Allocate(size, ptr);
        if (status.ok()) {
            stats_.current_allocated += size;
            stats_.peak_allocated = std::max(stats_.peak_allocated, stats_.current_allocated);
            stats_.num_allocations++;
            LOGI("MemoryPool::Allocate size={}", size);
        }
        return status;
    }
    auto status = large_.Allocate(size, ptr);
    if (status.ok()) {
        stats_.current_allocated += size;
        stats_.peak_allocated = std::max(stats_.peak_allocated, stats_.current_allocated);
        stats_.num_allocations++;
        LOGI("MemoryPool::Allocate size={}", size);
    }
    return status;
}

Status MemoryPool::Deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    Status status;
    if (size < small_pool_max_) {
        status = small_.Deallocate(ptr, size);
    } else {
        status = large_.Deallocate(ptr, size);
    }
    if (status.ok()) {
        stats_.current_allocated -= size;
        stats_.num_deallocations++;
        LOGI("MemoryPool::Deallocate size={}", size);
    }
    return status;
}

void MemoryPool::Trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    small_.Trim();
    large_.Trim();
}

PoolStats MemoryPool::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

// SmallPool
MemoryPool::SmallPool::SmallPool(Allocator* upstream, size_t alignment)
    : upstream_(upstream), alignment_(alignment) {}

int MemoryPool::SmallPool::BucketIndex(size_t size) const {
    if (size <= 64) return 0;
    int idx = 0;
    size_t s = 128;
    while (s < size && idx < kNumBuckets - 1) { s *= 2; idx++; }
    return idx;
}

size_t MemoryPool::SmallPool::BucketSize(int index) const {
    return std::max(static_cast<size_t>(64), static_cast<size_t>(128) << index);
}

Status MemoryPool::SmallPool::Allocate(size_t size, void** ptr) {
    int idx = BucketIndex(size);
    if (!free_list_[idx].empty()) {
        *ptr = free_list_[idx].back();
        free_list_[idx].pop_back();
        return Status::OK();
    }
    return upstream_->Allocate(ptr, BucketSize(idx));
}

Status MemoryPool::SmallPool::Deallocate(void* ptr, size_t size) {
    int idx = BucketIndex(size);
    if (free_list_[idx].size() < 64) {
        free_list_[idx].push_back(ptr);
        return Status::OK();
    }
    return upstream_->Deallocate(ptr);
}

void MemoryPool::SmallPool::Trim() {
    for (int i = 0; i < kNumBuckets; ++i) {
        for (auto* p : free_list_[i]) upstream_->Deallocate(p);
        free_list_[i].clear();
    }
}

// LargePool
MemoryPool::LargePool::LargePool(Allocator* upstream)
    : upstream_(upstream) {}

Status MemoryPool::LargePool::Allocate(size_t size, void** ptr) {
    return upstream_->Allocate(ptr, size);
}

Status MemoryPool::LargePool::Deallocate(void* ptr, size_t) {
    return upstream_->Deallocate(ptr);
}

void MemoryPool::LargePool::Trim() {}

} // namespace compute_engine
