#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/ops/op.h"
#include "ops/traditional/attrs_util.h"

namespace ie {
namespace detail {

struct PoolConfig {
    bool has_kernel_shape{false};
    std::array<int64_t, 2> kernel_shape{0, 0};
    std::array<int64_t, 2> strides{1, 1};
    std::array<int64_t, 4> pads{0, 0, 0, 0};
    AutoPad auto_pad{AutoPad::NotSet};
    bool ceil_mode{false};
    bool count_include_pad{false};
};

Status parse_pool_config(const Attrs& attrs, PoolConfig& config);

Status pool2d_forward(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs,
                      ExecutionContext& context, const PoolConfig& config,
                      bool max_pool);

Status global_avg_pool_forward(const std::vector<Tensor*>& inputs,
                               std::vector<Tensor*>& outputs,
                               ExecutionContext& context);

}  // namespace detail

// Max pooling over NCHW F32 tensors.
class MaxPool2DOp final : public Op {
public:
    Status configure(const Attrs& attrs) override {
        return detail::parse_pool_config(attrs, config_);
    }

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        return detail::pool2d_forward(inputs, outputs, context, config_, true);
    }

private:
    detail::PoolConfig config_{};
};

// Average pooling over NCHW F32 tensors.
class AvgPool2DOp final : public Op {
public:
    Status configure(const Attrs& attrs) override {
        return detail::parse_pool_config(attrs, config_);
    }

    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        return detail::pool2d_forward(inputs, outputs, context, config_, false);
    }

private:
    detail::PoolConfig config_{};
};

// Global average pooling: [N, C, H, W] -> [N, C, 1, 1].
class GlobalAvgPoolOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   ExecutionContext& context) override {
        return detail::global_avg_pool_forward(inputs, outputs, context);
    }
};

}  // namespace ie
