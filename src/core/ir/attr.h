#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ie {

// List alternatives are appended so existing alternatives keep their indices.
using AttrValue =
    std::variant<int64_t, double, bool, std::string, std::vector<int64_t>,
                 std::vector<double>>;
using Attrs = std::unordered_map<std::string, AttrValue>;

}  // namespace ie
