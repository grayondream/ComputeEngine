#pragma once

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/runtime/execution_context.h"
#include "core/scheduler/plan.h"
#include "core/scheduler/scheduler.h"
#include "hybrid/planner/hybrid_planner.h"

namespace ie {

// Hybrid scheduler surface (docs/readme.md §7.4). Batching and execution are
// not implemented in P0.5.
class HybridScheduler final : public Scheduler {
public:
    void submit(Request) override {}
    BatchPlan plan_next_iteration() override { return {}; }
    Status execute_iteration(const BatchPlan&) override {
        return Error::unsupported("hybrid execute_iteration not implemented in P0.5");
    }
    void on_request_complete(RequestId) override {}
    void preempt(RequestId) override {}

    Status execute(const Graph&, const ExecutionPlan&, ExecutionContext&) {
        return Error::unsupported("hybrid execute not implemented in P0.5");
    }
};

}  // namespace ie
