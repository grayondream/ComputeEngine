#pragma once

#include <string>
#include <vector>

namespace ie {

// Rules fired during one inference cycle (docs/readme.md §8.1).
struct Agenda {
    std::vector<std::string> fired_rules;
};

// Human-readable inference steps for rule-chain visualization.
struct Trace {
    std::vector<std::string> steps;
};

}  // namespace ie
