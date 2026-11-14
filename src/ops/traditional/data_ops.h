#pragma once

#include <cstdint>
#include <vector>

#include "core/ops/op.h"
#include "ops/traditional/attrs_util.h"

namespace ie {

// Concatenates F32 tensors along `axis`. Every dimension other than `axis` must
// match exactly; `axis` may be negative and defaults to 0.
class ConcatOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    int64_t axis_{0};
};

// Extracts a strided sub-tensor. Parameters come from the `starts/ends/axes/
// steps` attributes, or from inputs[1..4] (F32 values cast to int64) when the
// optional inputs are present. Negative axes, starts, and ends are normalized;
// out-of-range values are clamped. Negative steps are supported.
class SliceOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_starts_{false};
    bool has_ends_{false};
    bool has_axes_{false};
    bool has_steps_{false};
    std::vector<int64_t> starts_;
    std::vector<int64_t> ends_;
    std::vector<int64_t> axes_;
    std::vector<int64_t> steps_;
};

// Pads an F32 tensor with a constant value. Pads use the ONNX
// [begins..., ends...] layout and come from the `pads` attribute or inputs[1].
// The fill value comes from the `value` attribute or inputs[2]. Only
// mode="constant" is implemented; negative pads crop the input.
class PadOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_pads_{false};
    bool has_value_{false};
    std::vector<int64_t> pads_;
    double value_{0.0};
};

// Nearest-neighbor NCHW upsampling. Scales come from the `scales` attribute or
// inputs[1] and must contain one positive value per NCHW dimension.
class UpsampleOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_scales_{false};
    std::vector<double> scales_;
};

// Gathers slices of the first input along `axis` using F32 indices (cast to
// int64). Negative axes and indices are normalized.
class GatherOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    int64_t axis_{0};
};

// Reduces an F32 tensor over `axes` using either "mean" or "sum". An empty or
// missing axes list reduces every axis; `keepdims` defaults to 1.
class ReduceOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    enum class Mode { Mean, Sum };

    bool has_axes_{false};
    std::vector<int64_t> axes_;
    bool keepdims_{true};
    Mode mode_{Mode::Mean};
};

// Clamps an F32 tensor. Bounds come from the `min`/`max` attributes or
// inputs[1..2]; a missing bound is left unbounded.
class ClipOp final : public Op {
public:
    Status configure(const Attrs& attrs) override;

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override;

private:
    bool has_min_{false};
    bool has_max_{false};
    double min_{0.0};
    double max_{0.0};
};

}  // namespace ie
