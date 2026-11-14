#pragma once

#include <vector>

#include "core/ops/op.h"

namespace ie {

// Elementwise logistic sigmoid over an F32 tensor; the output shape is
// unchanged.
class SigmoidOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;
};

// Elementwise hyperbolic tangent over an F32 tensor; the output shape is
// unchanged.
class TanhOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;
};

}  // namespace ie
