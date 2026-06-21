#include "compute_engine/platform/profiler.h"
#include <chrono>
#include <sstream>
#include <thread>
#include "compute_engine/platform/logger.h"

namespace compute_engine {

Profiler& Profiler::Instance() {
    static Profiler instance;
    LOGI("Profiler initialized");
    return instance;
}

Profiler::Scope::Scope(const char* name)
    : name_(name),
      start_ns_(static_cast<uint64_t>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count())) {}

Profiler::Scope::~Scope() {
    auto end_ns = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    LOGT("Scope '{}' finished ({} ns)", name_, end_ns - start_ns_);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    Profiler::Instance().AddEvent(name_, start_ns_, end_ns);
}

void Profiler::AddEvent(const char* name, uint64_t start, uint64_t end) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (static_cast<int>(events_.size()) >= max_events_) return;
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    events_.push_back({name, start, end, static_cast<int>(tid)});
}

std::string Profiler::ExportChromeTrace() {
    std::lock_guard<std::mutex> lock(mutex_);
    LOGI("Exporting chrome trace ({} events)", events_.size());
    std::ostringstream ss;
    ss << "{\"traceEvents\":[";
    for (size_t i = 0; i < events_.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"name\":\"" << events_[i].name
           << "\",\"ph\":\"X\",\"ts\":" << events_[i].start_ns / 1000
           << ",\"dur\":" << (events_[i].end_ns - events_[i].start_ns) / 1000
           << ",\"tid\":" << events_[i].thread_id << "}";
    }
    ss << "]}";
    return ss.str();
}

void Profiler::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
}

void Profiler::SetMaxEvents(int n) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_events_ = n;
}

} // namespace compute_engine
