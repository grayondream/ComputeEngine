#pragma once

#include "core/ops/unimplemented_op.h"

namespace ie::experimental::attention {

// Flash-attention kernel (docs/readme.md §11.1). P0.5 freezes the surface only.
class FlashAttentionOp final : public UnimplementedOp {
public:
    FlashAttentionOp() : UnimplementedOp("FlashAttention") {}
};

}  // namespace ie::experimental::attention

namespace ie {
using experimental::attention::FlashAttentionOp;
}  // namespace ie
