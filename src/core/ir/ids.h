#pragma once

#include <cstdint>

namespace ie {

using NodeId = uint32_t;
using ValueId = uint32_t;

inline constexpr ValueId kInvalidValue = 0xFFFFFFFFu;
inline constexpr NodeId kInvalidNode = 0xFFFFFFFFu;

}  // namespace ie
