#include <gtest/gtest.h>

#include "ops/rule/rule_ops.h"
#include "rules/rule_engine.h"

using namespace ie;

TEST(RuleEngineTest, StoresFactsAndRulesButFireIsUnsupported) {
    RuleEngine engine;
    engine.add_fact(Fact{"person", {{"age", int64_t{20}}}});
    Rule rule;
    rule.name = "adult";
    rule.condition = [](const Fact&, const WorkingMemory&) { return true; };
    engine.add_rule(rule);
    auto agenda = engine.fire();
    ASSERT_FALSE(agenda.has_value());
    EXPECT_EQ(agenda.error().code, Error::Code::Unsupported);
    EXPECT_TRUE(engine.trace().steps.empty());
}

TEST(RuleOpsTest, DefaultsAreUnsupported) {
    RuleMatchOp match;
    RuleFireOp fire;
    FactAssertOp assert_fact;
    std::vector<Tensor*> in;
    std::vector<Tensor*> out;
    ExecutionContext context;
    EXPECT_EQ(match.forward(in, out, context).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(fire.forward(in, out, context).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(assert_fact.forward(in, out, context).error().code,
              Error::Code::Unsupported);
}
