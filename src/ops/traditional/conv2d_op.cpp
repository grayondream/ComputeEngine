#include "ops/traditional/conv2d_op.h"

#include <cstddef>
#include <utility>

namespace ie {
namespace detail {
namespace {

// Local reference row kernel used when the execution context has no injected
// matmul_row implementation. Rows are independent, so an injected kernel can
// only change the accumulation order inside one output element.
void conv_row_reference(const float* a_row, const float* b, float* out_row,
                        std::int64_t k, std::int64_t n) {
    for (std::int64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (std::int64_t t = 0; t < k; ++t) {
            acc += a_row[t] * b[t * n + j];
        }
        out_row[j] = acc;
    }
}

}  // namespace

Status parse_conv_config(const Attrs& attrs, ConvConfig& config) {
    config = ConvConfig{};
    if (auto status = read_pair(attrs, "strides", config.strides);
        !status.has_value()) {
        return status;
    }
    if (auto status = read_pair(attrs, "dilations", config.dilations);
        !status.has_value()) {
        return status;
    }
    if (auto status = read_pads(attrs, config.pads); !status.has_value()) {
        return status;
    }
    IntAttr group;
    if (auto status = read_int(attrs, "group", group); !status.has_value()) {
        return status;
    }
    if (group.found) {
        if (group.value <= 0) {
            return Error::invalid_argument("group must be positive");
        }
        config.group = group.value;
    }
    if (auto status = read_auto_pad(attrs, config.auto_pad);
        !status.has_value()) {
        return status;
    }
    IntListAttr kernel;
    if (auto status = read_int_list(attrs, "kernel_shape", kernel);
        !status.has_value()) {
        return status;
    }
    if (kernel.found) {
        if (kernel.values.size() != 2) {
            return Error::invalid_argument(
                "kernel_shape must contain 2 values");
        }
        if (kernel.values[0] <= 0 || kernel.values[1] <= 0) {
            return Error::invalid_argument(
                "kernel_shape values must be positive");
        }
        config.has_kernel_shape = true;
        config.kernel_shape = {kernel.values[0], kernel.values[1]};
    }
    return {};
}

Status conv2d_forward(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs,
                      ExecutionContext& context, const ConvConfig& config,
                      bool depthwise) {
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Conv2D expects two or three inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Conv2D requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& x = *inputs[0];
    const Tensor& weight = *inputs[1];
    if (x.shape.dims.size() != 4 || weight.shape.dims.size() != 4) {
        return Error::invalid_argument("Conv2D expects 4D NCHW tensors");
    }
    const std::int64_t n = x.shape.dims[0];
    const std::int64_t c = x.shape.dims[1];
    const std::int64_t h = x.shape.dims[2];
    const std::int64_t w = x.shape.dims[3];
    const std::int64_t m = weight.shape.dims[0];
    const std::int64_t channels_per_group = weight.shape.dims[1];
    const std::int64_t k_h = weight.shape.dims[2];
    const std::int64_t k_w = weight.shape.dims[3];
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || m <= 0 ||
        channels_per_group <= 0 || k_h <= 0 || k_w <= 0) {
        return Error::invalid_argument("Conv2D expects positive dimensions");
    }

    const std::int64_t group = config.group;
    if (group <= 0 || c % group != 0 || m % group != 0) {
        return Error::invalid_argument(
            "Conv2D group must divide input and output channels");
    }
    if (channels_per_group != c / group) {
        return Error::invalid_argument(
            "Conv2D weight channels do not match C/group");
    }
    if (depthwise && group != c) {
        return Error::invalid_argument(
            "DepthwiseConv2D requires group == input channels");
    }
    if (config.has_kernel_shape &&
        (config.kernel_shape[0] != k_h || config.kernel_shape[1] != k_w)) {
        return Error::invalid_argument(
            "kernel_shape does not match the weight tensor");
    }

    bool has_bias = false;
    const float* bias = nullptr;
    if (inputs.size() == 3) {
        const Tensor& bias_tensor = *inputs[2];
        if (bias_tensor.shape.dims.size() != 1 ||
            bias_tensor.shape.dims[0] != m) {
            return Error::invalid_argument("Conv2D bias must have shape [M]");
        }
        has_bias = true;
        bias = bias_tensor.data<float>();
    }

    std::array<std::int64_t, 4> pads = config.pads;
    if (config.auto_pad == AutoPad::Valid) {
        pads = {0, 0, 0, 0};
    } else if (config.auto_pad == AutoPad::SameUpper ||
               config.auto_pad == AutoPad::SameLower) {
        const bool lower = config.auto_pad == AutoPad::SameLower;
        const auto [top, bottom] = same_padding(
            h, k_h, config.dilations[0], config.strides[0], lower);
        const auto [left, right] = same_padding(
            w, k_w, config.dilations[1], config.strides[1], lower);
        pads = {top, left, bottom, right};
    }

    const std::int64_t effective_h =
        config.dilations[0] * (k_h - 1) + 1;
    const std::int64_t effective_w =
        config.dilations[1] * (k_w - 1) + 1;
    if (h + pads[0] + pads[2] < effective_h ||
        w + pads[1] + pads[3] < effective_w) {
        return Error::invalid_argument("Conv2D kernel exceeds padded input");
    }
    const std::int64_t out_h =
        (h + pads[0] + pads[2] - effective_h) / config.strides[0] + 1;
    const std::int64_t out_w =
        (w + pads[1] + pads[3] - effective_w) / config.strides[1] + 1;
    if (out_h <= 0 || out_w <= 0) {
        return Error::invalid_argument("Conv2D produces an empty output");
    }

    auto result = Tensor::allocate(Shape{{n, m, out_h, out_w}}, DType::F32,
                                   context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const std::int64_t spatial = out_h * out_w;
    const std::int64_t reduction = channels_per_group * k_h * k_w;
    const std::int64_t outputs_per_group = m / group;
    const std::int64_t channels_per_group_input = c / group;
    std::vector<float> columns(
        static_cast<std::size_t>(reduction * spatial));

    const KernelFns::MatMulRowFn row_fn =
        (context.kernels != nullptr && context.kernels->matmul_row != nullptr)
            ? context.kernels->matmul_row
            : conv_row_reference;

    const float* x_data = x.data<float>();
    const float* weight_data = weight.data<float>();
    float* out_data = result->data<float>();

    for (std::int64_t batch = 0; batch < n; ++batch) {
        for (std::int64_t g = 0; g < group; ++g) {
            // im2col for this group: reduction x spatial, zero padded.
            std::int64_t row = 0;
            for (std::int64_t c_local = 0; c_local < channels_per_group_input;
                 ++c_local) {
                const std::int64_t channel = g * channels_per_group_input +
                                             c_local;
                const float* x_channel =
                    x_data + (batch * c + channel) * h * w;
                for (std::int64_t ky = 0; ky < k_h; ++ky) {
                    for (std::int64_t kx = 0; kx < k_w; ++kx) {
                        float* column_row =
                            columns.data() + row * spatial;
                        for (std::int64_t oy = 0; oy < out_h; ++oy) {
                            const std::int64_t iy =
                                oy * config.strides[0] - pads[0] +
                                ky * config.dilations[0];
                            for (std::int64_t ox = 0; ox < out_w; ++ox) {
                                const std::int64_t ix =
                                    ox * config.strides[1] - pads[1] +
                                    kx * config.dilations[1];
                                float value = 0.0f;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                    value = x_channel[iy * w + ix];
                                }
                                column_row[oy * out_w + ox] = value;
                            }
                        }
                        ++row;
                    }
                }
            }

            for (std::int64_t m_local = 0; m_local < outputs_per_group;
                 ++m_local) {
                const std::int64_t channel_out = g * outputs_per_group +
                                                 m_local;
                const float* weight_row =
                    weight_data + channel_out * reduction;
                float* out_row =
                    out_data + (batch * m + channel_out) * spatial;
                row_fn(weight_row, columns.data(), out_row, reduction,
                       spatial);
                if (has_bias && bias[channel_out] != 0.0f) {
                    const float value = bias[channel_out];
                    for (std::int64_t p = 0; p < spatial; ++p) {
                        out_row[p] += value;
                    }
                }
            }
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace detail
}  // namespace ie
