#pragma once

#include "core/base/error.h"
#include "core/hardware/profile.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Plugin surface for probing new hardware (docs/readme.md §12.1).
class HardwareProbePlugin {
public:
    virtual ~HardwareProbePlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Result<HardwareProfile> probe() = 0;
};

}  // namespace ie
