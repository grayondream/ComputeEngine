#pragma once

#include <utility>
#include <vector>

#include "core/base/error.h"
#include "rules/agenda/agenda.h"
#include "rules/fact/fact.h"
#include "rules/rule/rule.h"

namespace ie {

// Forward-chaining rule engine (docs/readme.md §8.1). P0.5 freezes the
// contract only: facts and rules are stored, but fire() reports Unsupported.
class RuleEngine {
public:
    void add_fact(Fact fact) { facts_.facts.push_back(std::move(fact)); }

    void add_rule(Rule rule) { rules_.push_back(std::move(rule)); }

    Result<Agenda> fire() {
        return Error::unsupported("rule engine not implemented in P0.5");
    }

    const Trace& trace() const { return trace_; }

private:
    WorkingMemory facts_;
    std::vector<Rule> rules_;
    Trace trace_;
};

}  // namespace ie
