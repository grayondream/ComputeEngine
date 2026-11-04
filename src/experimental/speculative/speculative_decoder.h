#pragma once

#include <span>
#include <vector>

#include "core/base/error.h"

namespace ie::experimental::speculative {

// Speculative decoding contract (docs/readme.md §11.1). P0.5 freezes the
// surface only; draft-model verification lands later.
class SpeculativeDecoder {
public:
    struct Config {
        int draft_tokens{0};
    };

    Result<std::vector<int>> decode(std::span<const float> /*logits*/,
                                    const Config& /*config*/) {
        return Error::unsupported(
            "speculative decoding not implemented in P0.5");
    }
};

}  // namespace ie::experimental::speculative

namespace ie {
using experimental::speculative::SpeculativeDecoder;
}  // namespace ie
