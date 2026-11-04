#pragma once

#include <string>

#include "core/plugin/plugin_meta.h"

namespace ie {

// Learning-rate scheduler plugin (docs/readme.md §9.1).
class LRSchedulerPlugin {
public:
    virtual ~LRSchedulerPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::string scheduler_name() const = 0;
    virtual float get_lr(int step) = 0;
};

}  // namespace ie
