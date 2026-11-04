#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ie {

// Value carried by a fact field (docs/readme.md §8.1).
using FactValue = std::variant<int64_t, double, bool, std::string>;

// A single fact in working memory (docs/readme.md §8.1).
struct Fact {
    std::string type;
    std::unordered_map<std::string, FactValue> fields;
};

// Mutable set of facts seen by conditions and rule actions.
struct WorkingMemory {
    std::vector<Fact> facts;
};

}  // namespace ie
