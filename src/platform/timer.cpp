#include "compute_engine/platform/timer.h"
#include "compute_engine/platform/logger.h"
#include <chrono>

namespace compute_engine {

Timer::Timer() {
    Reset();
}

void Timer::Reset() {
    start_ns_ = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

uint64_t Timer::ElapsedNs() const {
    auto now = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return now - start_ns_;
}

double Timer::ElapsedMs() const { return ElapsedNs() / 1e6; }
double Timer::ElapsedUs() const { return ElapsedNs() / 1e3; }
double Timer::ElapsedSec() const { return ElapsedNs() / 1e9; }

} // namespace compute_engine
