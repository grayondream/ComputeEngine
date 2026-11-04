#pragma once

#include "core/base/error.h"
#include "core/base/types.h"

namespace ie {

// Automatic mixed precision configuration. Disabled by default (P0.5).
struct AmpConfig {
    bool enabled{false};
    DType compute_dtype{DType::F16};
    float loss_scale{1.0f};
};

class AmpManager {
public:
    Status enable(const AmpConfig& config) {
        (void)config;
        return Error::unsupported("amp not implemented in P0.5");
    }

    bool is_enabled() const { return enabled_; }

private:
    bool enabled_{false};
};

}  // namespace ie
