#pragma once

#include <cstdint>
#include <vector>

namespace ie {

using BlockId = std::uint32_t;
using BlockHash = std::uint64_t;
using TensorHash = std::uint64_t;

struct BlockTable {
    std::vector<BlockId> blocks;
};

enum class QuantFormat { NONE, INT8, FP8 };

}  // namespace ie
