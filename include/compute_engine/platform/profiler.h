#pragma once

#include "compute_engine/core/macros.h"
#include <vector>
#include <mutex>
#include <string>
#include <cstdint>

namespace compute_engine {

class Profiler {
public:
    static Profiler& Instance();

    struct Scope {
        Scope(const char* name);
        ~Scope();
        const char* name_;
        uint64_t start_ns_;
    };

    struct Event {
        const char* name;
        uint64_t start_ns;
        uint64_t end_ns;
        int thread_id;
    };

    void AddEvent(const char* name, uint64_t start, uint64_t end);
    std::string ExportChromeTrace();
    void Clear();
    void SetMaxEvents(int n);

private:
    Profiler() = default;
    std::vector<Event> events_;
    int max_events_ = 100000;
    std::mutex mutex_;
};

#define CE_PROFILE(name) ::compute_engine::Profiler::Scope CE_CONCAT(_profile_, __COUNTER__)(name)

} // namespace compute_engine
