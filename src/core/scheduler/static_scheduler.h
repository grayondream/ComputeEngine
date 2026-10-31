#pragma once

#include <functional>
#include <vector>

#include "core/scheduler/scheduler.h"

namespace ie {

// P0 static scheduler: one prefill per pending request, no decode.
class StaticScheduler final : public Scheduler {
public:
    using ExecutorFn = std::function<Status(const BatchPlan&)>;

    explicit StaticScheduler(ExecutorFn executor = {})
        : executor_(std::move(executor)) {}

    void submit(Request request) override;
    BatchPlan plan_next_iteration() override;
    Status execute_iteration(const BatchPlan& plan) override;
    void on_request_complete(RequestId id) override;
    void preempt(RequestId id) override;

private:
    std::vector<Request> pending_;
    ExecutorFn executor_;
};

}  // namespace ie
