#pragma once

#include "core/backend/engine_config.h"

namespace ie {

class DeviceSelector {
public:
    // P0 has no GPU backend; the selector always reports NONE.
    static GpuBackend select() { return GpuBackend::NONE; }
};

}  // namespace ie
