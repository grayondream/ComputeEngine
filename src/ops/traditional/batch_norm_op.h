#pragma once

#include <vector>

#include "core/ops/op.h"

namespace ie {

// Batch normalization over the channel dimension (NCHW). Inputs are
// [x, scale, bias, mean, var]; output has the same shape as x.
class BatchNormOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    double epsilon_{1e-5};
};

}  // namespace ie
