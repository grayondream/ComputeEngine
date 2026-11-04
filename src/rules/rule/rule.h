#pragma once

#include <functional>
#include <string>

#include "rules/fact/fact.h"

namespace ie {

// Predicate evaluated against a fact and the current working memory.
using Condition = std::function<bool(const Fact&, const WorkingMemory&)>;

// Production rule (docs/readme.md §8.1). P0.5 only freezes the shape.
struct Rule {
    std::string name;
    Condition condition;
    std::function<void(WorkingMemory&)> action;
    int priority{0};
};

}  // namespace ie
