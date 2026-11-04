#pragma once

#include <memory>

#include "core/base/error.h"
#include "core/ir/attr.h"
#include "core/ops/op.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Plugin surface for registering op implementations (docs/readme.md §6.5).
class OpPlugin {
public:
    virtual ~OpPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Result<std::unique_ptr<Op>> create(const Attrs& attrs) = 0;
    virtual bool is_control_flow() const { return false; }
};

}  // namespace ie
