#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

namespace ie {

using AttrValue = std::variant<int64_t, double, bool, std::string>;
using Attrs = std::unordered_map<std::string, AttrValue>;

}  // namespace ie
