#pragma once

#include "core/base/error.h"
#include "core/scheduler/plan.h"

namespace ie {

class Scheduler {
public:
    virtual ~Scheduler() = default;

    virtual void submit(Request request) = 0;
    virtual BatchPlan plan_next_iteration() = 0;
    virtual Status execute_iteration(const BatchPlan& plan) = 0;
    virtual void on_request_complete(RequestId id) = 0;
    virtual void preempt(RequestId id) = 0;
};

}  // namespace ie
