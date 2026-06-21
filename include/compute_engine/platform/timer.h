#pragma once

#include "compute_engine/core/macros.h"
#include <cstdint>

namespace compute_engine {

class Timer {
public:
    Timer();

    void Reset();
    uint64_t ElapsedNs() const;
    double ElapsedMs() const;
    double ElapsedUs() const;
    double ElapsedSec() const;

private:
    uint64_t start_ns_;
};

} // namespace compute_engine
