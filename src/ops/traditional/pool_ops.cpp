#include "ops/traditional/pool_ops.h"

#include <cstddef>
#include <limits>
#include <utility>

namespace ie {
namespace detail {
namespace {

Status read_optional_flag(const Attrs& attrs, const char* key, bool& out) {
    out = false;
    IntAttr flag;
    if (auto status = read_int(attrs, key, flag); !status.has_value()) {
        return status;
    }
    if (flag.found) {
        out = flag.value != 0;
    }
    return {};
}

// Resolves the pads attribute (or auto_pad) and validates the window shape.
Status resolve_pool_geometry(const PoolConfig& config, std::int64_t input_h,
                             std::int64_t input_w, std::int64_t kernel_h,
                             std::int64_t kernel_w, std::array<int64_t, 4>& pads,
                             std::int64_t& output_h, std::int64_t& output_w) {
    pads = config.pads;
    if (config.auto_pad == AutoPad::Valid) {
        pads = {0, 0, 0, 0};
    } else if (config.auto_pad == AutoPad::SameUpper ||
               config.auto_pad == AutoPad::SameLower) {
        const bool lower = config.auto_pad == AutoPad::SameLower;
        const auto [top, bottom] =
            same_padding(input_h, kernel_h, 1, config.strides[0], lower);
        const auto [left, right] =
            same_padding(input_w, kernel_w, 1, config.strides[1], lower);
        pads = {top, left, bottom, right};
    }

    const std::int64_t span_h = input_h + pads[0] + pads[2];
    const std::int64_t span_w = input_w + pads[1] + pads[3];
    if (span_h < kernel_h || span_w < kernel_w) {
        return Error::invalid_argument("Pool kernel exceeds padded input");
    }
    if (config.ceil_mode) {
        output_h = ceil_div(span_h - kernel_h, config.strides[0]) + 1;
        output_w = ceil_div(span_w - kernel_w, config.strides[1]) + 1;
        // Drop a trailing window that would start inside the padding region.
        if ((output_h - 1) * config.strides[0] >= input_h + pads[0]) {
            --output_h;
        }
        if ((output_w - 1) * config.strides[1] >= input_w + pads[1]) {
            --output_w;
        }
    } else {
        output_h = (span_h - kernel_h) / config.strides[0] + 1;
        output_w = (span_w - kernel_w) / config.strides[1] + 1;
    }
    if (output_h <= 0 || output_w <= 0) {
        return Error::invalid_argument("Pool produces an empty output");
    }
    return {};
}

}  // namespace

Status parse_pool_config(const Attrs& attrs, PoolConfig& config) {
    config = PoolConfig{};
    IntListAttr kernel;
    if (auto status = read_int_list(attrs, "kernel_shape", kernel);
        !status.has_value()) {
        return status;
    }
    if (!kernel.found || kernel.values.empty()) {
        return Error::invalid_argument("kernel_shape is required");
    }
    if (kernel.values.size() != 2) {
        return Error::invalid_argument("kernel_shape must contain 2 values");
    }
    if (kernel.values[0] <= 0 || kernel.values[1] <= 0) {
        return Error::invalid_argument(
            "kernel_shape values must be positive");
    }
    config.has_kernel_shape = true;
    config.kernel_shape = {kernel.values[0], kernel.values[1]};

    if (auto status = read_pair(attrs, "strides", config.strides);
        !status.has_value()) {
        return status;
    }
    if (auto status = read_pads(attrs, config.pads); !status.has_value()) {
        return status;
    }
    if (auto status = read_auto_pad(attrs, config.auto_pad);
        !status.has_value()) {
        return status;
    }
    if (auto status = read_optional_flag(attrs, "ceil_mode", config.ceil_mode);
        !status.has_value()) {
        return status;
    }
    if (auto status = read_optional_flag(attrs, "count_include_pad",
                                         config.count_include_pad);
        !status.has_value()) {
        return status;
    }
    return {};
}

Status pool2d_forward(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs,
                      ExecutionContext& context, const PoolConfig& config,
                      bool max_pool) {
    if (inputs.size() != 1 || outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Pooling expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Pooling requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }
    if (!config.has_kernel_shape) {
        return Error::invalid_argument("kernel_shape is required");
    }

    const Tensor& x = *inputs[0];
    if (x.shape.dims.size() != 4) {
        return Error::invalid_argument("Pooling expects 4D NCHW tensors");
    }
    const std::int64_t n = x.shape.dims[0];
    const std::int64_t c = x.shape.dims[1];
    const std::int64_t h = x.shape.dims[2];
    const std::int64_t w = x.shape.dims[3];
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return Error::invalid_argument("Pooling expects positive dimensions");
    }
    const std::int64_t kernel_h = config.kernel_shape[0];
    const std::int64_t kernel_w = config.kernel_shape[1];
    if (kernel_h <= 0 || kernel_w <= 0) {
        return Error::invalid_argument(
            "kernel_shape values must be positive");
    }

    std::array<std::int64_t, 4> pads{0, 0, 0, 0};
    std::int64_t out_h = 0;
    std::int64_t out_w = 0;
    if (auto status = resolve_pool_geometry(config, h, w, kernel_h, kernel_w,
                                            pads, out_h, out_w);
        !status.has_value()) {
        return status;
    }

    auto result = Tensor::allocate(Shape{{n, c, out_h, out_w}}, DType::F32,
                                   context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in = x.data<float>();
    float* out = result->data<float>();
    constexpr float kNegativeInfinity =
        -std::numeric_limits<float>::infinity();
    for (std::int64_t batch = 0; batch < n; ++batch) {
        for (std::int64_t channel = 0; channel < c; ++channel) {
            const float* in_plane = in + (batch * c + channel) * h * w;
            float* out_plane = out + (batch * c + channel) * out_h * out_w;
            for (std::int64_t oy = 0; oy < out_h; ++oy) {
                for (std::int64_t ox = 0; ox < out_w; ++ox) {
                    float accumulator = max_pool ? kNegativeInfinity : 0.0f;
                    std::int64_t count = 0;
                    for (std::int64_t ky = 0; ky < kernel_h; ++ky) {
                        const std::int64_t iy =
                            oy * config.strides[0] - pads[0] + ky;
                        for (std::int64_t kx = 0; kx < kernel_w; ++kx) {
                            const std::int64_t ix =
                                ox * config.strides[1] - pads[1] + kx;
                            const bool valid =
                                iy >= 0 && iy < h && ix >= 0 && ix < w;
                            if (valid) {
                                const float value = in_plane[iy * w + ix];
                                if (max_pool) {
                                    accumulator =
                                        value > accumulator ? value
                                                            : accumulator;
                                } else {
                                    accumulator += value;
                                    if (!config.count_include_pad) {
                                        ++count;
                                    }
                                }
                            }
                            if (!max_pool && config.count_include_pad &&
                                iy >= -pads[0] && iy < h + pads[2] &&
                                ix >= -pads[1] && ix < w + pads[3]) {
                                ++count;
                            }
                        }
                    }
                    if (max_pool) {
                        out_plane[oy * out_w + ox] = accumulator;
                    } else {
                        out_plane[oy * out_w + ox] =
                            count > 0
                                ? accumulator /
                                      static_cast<float>(count)
                                : 0.0f;
                    }
                }
            }
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status global_avg_pool_forward(const std::vector<Tensor*>& inputs,
                               std::vector<Tensor*>& outputs,
                               ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "GlobalAvgPool expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("GlobalAvgPool requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& x = *inputs[0];
    if (x.shape.dims.size() != 4) {
        return Error::invalid_argument(
            "GlobalAvgPool expects 4D NCHW tensors");
    }
    const std::int64_t n = x.shape.dims[0];
    const std::int64_t c = x.shape.dims[1];
    const std::int64_t h = x.shape.dims[2];
    const std::int64_t w = x.shape.dims[3];
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return Error::invalid_argument(
            "GlobalAvgPool expects positive dimensions");
    }

    auto result = Tensor::allocate(Shape{{n, c, 1, 1}}, DType::F32,
                                   context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in = x.data<float>();
    float* out = result->data<float>();
    const std::int64_t spatial = h * w;
    for (std::int64_t batch = 0; batch < n; ++batch) {
        for (std::int64_t channel = 0; channel < c; ++channel) {
            const float* plane = in + (batch * c + channel) * spatial;
            float sum = 0.0f;
            for (std::int64_t i = 0; i < spatial; ++i) {
                sum += plane[i];
            }
            out[batch * c + channel] =
                sum / static_cast<float>(spatial);
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace detail
}  // namespace ie
