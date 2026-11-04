#pragma once

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/runtime/execution_context.h"
#include "core/scheduler/plan.h"

namespace ie {

// Iterative execution entry point for Transformer-style models
// (docs/readme.md §3.2). Implementations arrive in later milestones.
class IterativeExecutor {
public:
    virtual ~IterativeExecutor() = default;

    virtual Status run(const Graph& graph, const BatchPlan& plan,
                       ExecutionContext& context) = 0;
};

}  // namespace ie
