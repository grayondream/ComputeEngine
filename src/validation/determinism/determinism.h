#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/base/error.h"

namespace ie {

// docs/readme.md §10.4.
struct DeterminismConfig {
    bool enabled{false};
    bool deterministic_kernels{false};
    bool batch_invariant{false};
    bool disable_tf32{true};
    bool use_ieee_rounding{true};
    bool disable_prefix_caching{true};
};

struct BatchInvarianceReport {
    bool passed{false};
    std::size_t batches_checked{0};
    double max_abs_diff{0.0};
    std::string first_divergence;
};

class DeterminismGuard {
public:
    Status enable(const DeterminismConfig& config) {
        (void)config;
        return Error::unsupported("determinism guard not implemented in P0.5");
    }

    Result<BatchInvarianceReport> verify_batch_invariance(
        const std::string& prompt, std::int64_t seed = 42) {
        (void)prompt;
        (void)seed;
        return Error::unsupported(
            "batch invariance verification not implemented in P0.5");
    }

    Status warmup() {
        return Error::unsupported("determinism warmup not implemented in P0.5");
    }
};

}  // namespace ie
