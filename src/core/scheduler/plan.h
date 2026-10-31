#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/memory/allocator.h"
#include "core/memory/tensor.h"

namespace ie {

using RequestId = std::uint64_t;

struct Request {
    RequestId id{0};
    std::string model;
    std::vector<Tensor> inputs;
};

struct PrefillItem {
    RequestId request{0};
    std::vector<Tensor> inputs;
};

struct DecodeItem {
    RequestId request{0};
    std::size_t num_tokens{1};
};

struct TransferItem {
    BufferHandle src{nullptr};
    BufferHandle dst{nullptr};
    std::size_t bytes{0};
};

// Unified batch plan (docs/readme.md §3.2): traditional models use prefills
// once; Transformer models use prefills + decodes iteratively.
struct BatchPlan {
    std::vector<PrefillItem> prefills;
    std::vector<DecodeItem> decodes;
    std::vector<TransferItem> transfers;
    std::size_t token_budget{0};
};

}  // namespace ie
