#pragma once

#include <memory>

#include "core/ir/pass.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Plugin surface for compiler passes, consumed by PluginRegistry<PassPlugin>.
class PassPlugin {
public:
    virtual ~PassPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::unique_ptr<Pass> create() = 0;
};

}  // namespace ie
