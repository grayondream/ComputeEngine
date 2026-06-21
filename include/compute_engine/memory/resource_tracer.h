#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace compute_engine {

enum class ResourceType : uint8_t {
    TENSOR, BACKEND, STREAM, MEMORY_BLOCK, EVENT,
    SESSION, MODEL, KERNEL, DESCRIPTOR, SHADER, CONTEXT,
};

struct ResourceEntry {
    void* ptr;
    ResourceType type;
    const char* file;
    int line;
    std::string name;
};

class ResourceTracer {
public:
    static ResourceTracer& Instance();

    void TrackAlloc(void* ptr, ResourceType type,
                    const char* file, int line,
                    const std::string& name = "");

    bool TrackFree(void* ptr);

    void DumpLeaks();

    int64_t NumLive() const;

private:
    ResourceTracer() = default;
    ~ResourceTracer() = default;
    ResourceTracer(const ResourceTracer&) = delete;
    ResourceTracer& operator=(const ResourceTracer&) = delete;

    std::unordered_map<void*, ResourceEntry> live_map_;
    mutable std::mutex mutex_;
};

} // namespace compute_engine
