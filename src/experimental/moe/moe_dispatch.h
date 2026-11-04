#pragma once

#include "core/ops/unimplemented_op.h"

namespace ie::experimental::moe {

// MoE expert dispatch (docs/readme.md §11.2). P0.5 freezes the surface only.
class MoEDispatchOp final : public UnimplementedOp {
public:
    MoEDispatchOp() : UnimplementedOp("MoEDispatch") {}
};

}  // namespace ie::experimental::moe

namespace ie {
using experimental::moe::MoEDispatchOp;
}  // namespace ie
