#include "compute_engine/memory/resource_tracer.h"
#include "compute_engine/platform/logger.h"
#include <cstdio>

namespace compute_engine {

ResourceTracer& ResourceTracer::Instance() {
    static ResourceTracer instance;
    LOGI("ResourceTracer initialized");
    return instance;
}

void ResourceTracer::TrackAlloc(void* ptr, ResourceType type,
                                 const char* file, int line,
                                 const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_map_[ptr] = {ptr, type, file, line, name};
}

bool ResourceTracer::TrackFree(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_map_.erase(ptr) > 0;
}

void ResourceTracer::DumpLeaks() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (live_map_.empty()) return;
    LOGW("Resource Leak Detected: {} resources still alive!", live_map_.size());
    for (auto& [ptr, entry] : live_map_) {
        LOGW("  ptr={} type={} file={}:{} name={}", static_cast<const void*>(ptr),
             static_cast<int>(entry.type), entry.file, entry.line, entry.name);
    }
}

int64_t ResourceTracer::NumLive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int64_t>(live_map_.size());
}

} // namespace compute_engine
