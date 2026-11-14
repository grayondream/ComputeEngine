#pragma once

#include <cstdint>

#include "core/ops/op.h"
#include "ops/traditional/attrs_util.h"

namespace ie {

// General matrix multiply: Y = alpha * A' * B' + beta * C (ONNX Gemm).
// A and B are 2D F32 tensors; transA/transB select the transposed view.
// alpha/beta default to 1. C is optional and broadcasts as scalar, [N], [M],
// [M,1], [1,N], or [M,N].
class GemmOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    double alpha_{1.0};
    double beta_{1.0};
    int64_t trans_a_{0};
    int64_t trans_b_{0};
};

}  // namespace ie
