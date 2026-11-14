#pragma once

#include <cstdint>
#include <vector>

#include "core/ops/op.h"
#include "ops/traditional/attrs_util.h"

namespace ie {

// Flattens all dimensions before/after `axis` into a rank-2 tensor.
// `axis` defaults to 1 and may be negative (counted from the back).
class FlattenOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    int64_t axis_{1};
};

// Reshapes an F32 tensor without changing its element order. The target shape
// comes from inputs[1] (F32 values cast to int64) when present, otherwise from
// the `shape` int-list attribute. 0 copies the matching input dimension and one
// -1 is inferred.
class ReshapeOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_shape_attr_{false};
    std::vector<int64_t> shape_attr_;
};

// Permutes tensor axes. `perm` must be a permutation of [0, rank); when it is
// absent every axis is reversed.
class TransposeOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_perm_{false};
    std::vector<int64_t> perm_;
};

}  // namespace ie
