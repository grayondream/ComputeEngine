#include <gtest/gtest.h>

#include "core/scheduler/static_scheduler.h"

using namespace ie;

TEST(StaticSchedulerTest, PlanContainsOnePrefillPerRequest) {
    StaticScheduler s;
    s.submit(Request{1, "m", {Tensor::from_data({1, 2})}});
    s.submit(Request{2, "m", {Tensor::from_data({3, 4})}});
    auto plan = s.plan_next_iteration();
    ASSERT_EQ(plan.prefills.size(), 2u);
    EXPECT_TRUE(plan.decodes.empty());
}

TEST(StaticSchedulerTest, ExecuteIterationInvokesExecutor) {
    int calls = 0;
    StaticScheduler s([&](const BatchPlan&) {
        ++calls;
        return Status{};
    });
    s.submit(Request{7, "m", {}});
    auto plan = s.plan_next_iteration();
    ASSERT_TRUE(s.execute_iteration(plan).has_value());
    EXPECT_EQ(calls, 1);
    s.on_request_complete(7);
    EXPECT_TRUE(s.plan_next_iteration().prefills.empty());
}

TEST(StaticSchedulerTest, PreemptRemovesPendingRequest) {
    StaticScheduler s;
    s.submit(Request{9, "m", {}});
    s.preempt(9);
    EXPECT_TRUE(s.plan_next_iteration().prefills.empty());
}
