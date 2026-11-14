#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/ops/op.h"
#include "ops/traditional/attrs_util.h"

namespace ie {
namespace detail {

struct ConvConfig {
    std::array<int64_t, 2> strides{1, 1};
    std::array<int64_t, 2> dilations{1, 1};
    std::array<int64_t, 4> pads{0, 0, 0, 0};
    int64_t group{1};
    AutoPad auto_pad{AutoPad::NotSet};
    bool has_kernel_shape{false};
    std::array<int64_t, 2> kernel_shape{0, 0};
};

Status parse_conv_config(const Attrs& attrs, ConvConfig& config);

// Shared NCHW F32 convolution. `depthwise` additionally requires group to equal
// the input channel count; the math is the generic grouped convolution.
Status conv2d_forward(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs,
                      ExecutionContext& context, const ConvConfig& config,
                      bool depthwise);

}  // namespace detail

// 2D convolution with im2col + row GEMM. Inputs are [x, weight] or
// [x, weight, bias] in NCHW layout; weight is [M, C/group, kH, kW].
class Conv2DOp final : public Op {
public:
    Status configure(const Attrs& attrs) override {
        return detail::parse_conv_config(attrs, config_);
    }

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        return detail::conv2d_forward(inputs, outputs, context, config_, false);
    }

private:
    detail::ConvConfig config_{};
};

// Depthwise convolution: Conv2D with group == input channels.
class DepthwiseConv2DOp final : public Op {
public:
    Status configure(const Attrs& attrs) override {
        return detail::parse_conv_config(attrs, config_);
    }

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        return detail::conv2d_forward(inputs, outputs, context, config_, true);
    }

private:
    detail::ConvConfig config_{};
};

}  // namespace ie
