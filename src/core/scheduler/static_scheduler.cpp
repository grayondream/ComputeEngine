#include "core/scheduler/static_scheduler.h"

#include <algorithm>

namespace ie {

void StaticScheduler::submit(Request request) {
    pending_.push_back(std::move(request));
}

BatchPlan StaticScheduler::plan_next_iteration() {
    BatchPlan plan;
    for (const Request& request : pending_) {
        plan.prefills.push_back(PrefillItem{request.id, request.inputs});
        for (const Tensor& input : request.inputs) {
            plan.token_budget += static_cast<std::size_t>(input.numel());
        }
    }
    return plan;
}

Status StaticScheduler::execute_iteration(const BatchPlan& plan) {
    if (!executor_) {
        return {};
    }
    return executor_(plan);
}

void StaticScheduler::on_request_complete(RequestId id) {
    pending_.erase(
        std::remove_if(pending_.begin(), pending_.end(),
                       [id](const Request& request) { return request.id == id; }),
        pending_.end());
}

void StaticScheduler::preempt(RequestId id) { on_request_complete(id); }

}  // namespace ie
