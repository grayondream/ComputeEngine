#pragma once

#include "core/base/error.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/ir/model_kind.h"
#include "core/plugin/plugin_meta.h"
#include "core/scheduler/plan.h"

namespace ie {

struct SchedulerContext {
    HardwareProfile hardware;
    ModelKind model_kind{ModelKind::UNKNOWN};
};

// Plugin surface for custom scheduling policies (docs/readme.md §12.1).
class SchedulerPlugin {
public:
    virtual ~SchedulerPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Result<BatchPlan> plan(Graph& graph, SchedulerContext& context) = 0;
};

}  // namespace ie
