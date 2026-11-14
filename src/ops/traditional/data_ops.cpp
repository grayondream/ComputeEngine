#include "ops/traditional/data_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ie {
namespace {

bool checked_mul(int64_t lhs, int64_t rhs, int64_t& out) {
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checked_add(int64_t lhs, int64_t rhs, int64_t& out) {
    if (rhs > 0) {
        if (lhs > std::numeric_limits<int64_t>::max() - rhs) {
            return false;
        }
    } else if (lhs < std::numeric_limits<int64_t>::min() - rhs) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

Status checked_product(const std::vector<int64_t>& dims, std::size_t begin,
                       std::size_t end, int64_t& out) {
    int64_t product = 1;
    for (std::size_t i = begin; i < end; ++i) {
        if (!checked_mul(product, dims[i], product)) {
            return Error::invalid_argument("shape product overflows int64");
        }
    }
    out = product;
    return {};
}

// Shape/index parameters are carried as F32 by the loader. Reject values that
// cannot be represented as int64 before casting to avoid undefined behavior.
bool float_to_int64(float value, int64_t& out) {
    if (!std::isfinite(value)) {
        return false;
    }
    constexpr double kLimit = 9.2e18;
    const double wide = static_cast<double>(value);
    if (wide > kLimit || wide < -kLimit) {
        return false;
    }
    out = static_cast<int64_t>(value);
    return true;
}

Status tensor_to_int64(const Tensor& tensor, std::vector<int64_t>& out) {
    const int64_t count = tensor.shape.numel();
    if (count <= 0) {
        return Error::invalid_argument(
            "integer parameter tensor must not be empty");
    }
    out.resize(static_cast<std::size_t>(count));
    const float* data = tensor.data<float>();
    for (int64_t i = 0; i < count; ++i) {
        if (!float_to_int64(data[i], out[static_cast<std::size_t>(i)])) {
            return Error::invalid_argument(
                "integer parameter tensor contains an invalid value");
        }
    }
    return {};
}

Status read_numeric_list(const Attrs& attrs, const char* key, bool& found,
                         std::vector<double>& out) {
    found = false;
    out.clear();
    const auto it = attrs.find(key);
    if (it == attrs.end()) {
        return {};
    }
    if (const auto* values = std::get_if<std::vector<double>>(&it->second)) {
        found = true;
        out = *values;
        return {};
    }
    if (const auto* values = std::get_if<std::vector<int64_t>>(&it->second)) {
        found = true;
        out.reserve(values->size());
        for (int64_t value : *values) {
            out.push_back(static_cast<double>(value));
        }
        return {};
    }
    return Error::invalid_argument(std::string("attribute '") + key +
                                   "' must be a numeric list");
}

Status resolve_axis(int64_t axis, int64_t rank, int64_t& normalized) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return Error::invalid_argument("axis is out of range");
    }
    normalized = axis;
    return {};
}

Status require_positive_dims(const std::vector<int64_t>& dims,
                             const char* op_name) {
    if (dims.empty()) {
        return Error::invalid_argument(std::string(op_name) +
                                       " does not support scalars");
    }
    for (int64_t dim : dims) {
        if (dim <= 0) {
            return Error::invalid_argument(
                std::string(op_name) + " requires concrete input dimensions");
        }
    }
    return {};
}

}  // namespace

Status ConcatOp::configure(const Attrs& attrs) {
    axis_ = 0;
    detail::IntAttr axis;
    if (auto status = detail::read_int(attrs, "axis", axis);
        !status.has_value()) {
        return status;
    }
    if (axis.found) {
        axis_ = axis.value;
    }
    return {};
}

Status ConcatOp::forward(const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& outputs,
                         ExecutionContext& context) {
    if (inputs.empty() || outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Concat expects at least one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Concat requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const std::vector<int64_t>& first_dims = inputs[0]->shape.dims;
    const int64_t rank = static_cast<int64_t>(first_dims.size());
    if (rank < 1) {
        return Error::invalid_argument("Concat does not support scalars");
    }

    int64_t axis = 0;
    if (auto status = resolve_axis(axis_, rank, axis); !status.has_value()) {
        return status;
    }

    int64_t total_axis = 0;
    for (const Tensor* tensor : inputs) {
        const std::vector<int64_t>& dims = tensor->shape.dims;
        if (static_cast<int64_t>(dims.size()) != rank) {
            return Error::invalid_argument(
                "Concat requires inputs of the same rank");
        }
        for (int64_t d = 0; d < rank; ++d) {
            if (d != axis && dims[static_cast<std::size_t>(d)] !=
                                first_dims[static_cast<std::size_t>(d)]) {
                return Error::invalid_argument(
                    "Concat requires matching non-axis dimensions");
            }
        }
        if (dims[static_cast<std::size_t>(axis)] <= 0) {
            return Error::invalid_argument(
                "Concat requires concrete input dimensions");
        }
        if (!checked_add(total_axis, dims[static_cast<std::size_t>(axis)],
                         total_axis)) {
            return Error::invalid_argument(
                "Concat axis size overflows int64");
        }
    }

    std::vector<int64_t> out_dims = first_dims;
    out_dims[static_cast<std::size_t>(axis)] = total_axis;
    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Concat requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    int64_t outer = 1;
    int64_t inner = 1;
    if (auto status = checked_product(first_dims, 0,
                                      static_cast<std::size_t>(axis), outer);
        !status.has_value()) {
        return status;
    }
    if (auto status = checked_product(first_dims,
                                      static_cast<std::size_t>(axis) + 1,
                                      static_cast<std::size_t>(rank), inner);
        !status.has_value()) {
        return status;
    }

    float* out_data = result->data<float>();
    for (int64_t o = 0; o < outer; ++o) {
        int64_t axis_offset = 0;
        for (const Tensor* tensor : inputs) {
            const int64_t axis_dim =
                tensor->shape.dims[static_cast<std::size_t>(axis)];
            const int64_t block = axis_dim * inner;
            const float* source = tensor->data<float>() + o * block;
            float* destination = out_data + (o * total_axis + axis_offset) *
                                                inner;
            std::copy(source, source + block, destination);
            axis_offset += axis_dim;
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status SliceOp::configure(const Attrs& attrs) {
    has_starts_ = false;
    has_ends_ = false;
    has_axes_ = false;
    has_steps_ = false;
    starts_.clear();
    ends_.clear();
    axes_.clear();
    steps_.clear();

    detail::IntListAttr starts;
    if (auto status = detail::read_int_list(attrs, "starts", starts);
        !status.has_value()) {
        return status;
    }
    has_starts_ = starts.found;
    starts_ = std::move(starts.values);

    detail::IntListAttr ends;
    if (auto status = detail::read_int_list(attrs, "ends", ends);
        !status.has_value()) {
        return status;
    }
    has_ends_ = ends.found;
    ends_ = std::move(ends.values);

    detail::IntListAttr axes;
    if (auto status = detail::read_int_list(attrs, "axes", axes);
        !status.has_value()) {
        return status;
    }
    has_axes_ = axes.found;
    axes_ = std::move(axes.values);

    detail::IntListAttr steps;
    if (auto status = detail::read_int_list(attrs, "steps", steps);
        !status.has_value()) {
        return status;
    }
    has_steps_ = steps.found;
    steps_ = std::move(steps.values);
    return {};
}

Status SliceOp::forward(const std::vector<Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ExecutionContext& context) {
    if (inputs.empty() || inputs.size() == 2 || inputs.size() > 5 ||
        outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Slice expects 1 or 3..5 inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Slice requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (auto valid = require_positive_dims(dims, "Slice");
        !valid.has_value()) {
        return valid;
    }

    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> axes;
    std::vector<int64_t> steps;
    if (inputs.size() >= 3) {
        if (auto status = tensor_to_int64(*inputs[1], starts);
            !status.has_value()) {
            return status;
        }
        if (auto status = tensor_to_int64(*inputs[2], ends);
            !status.has_value()) {
            return status;
        }
        if (inputs.size() >= 4) {
            if (auto status = tensor_to_int64(*inputs[3], axes);
                !status.has_value()) {
                return status;
            }
        } else if (has_axes_) {
            axes = axes_;
        }
        if (inputs.size() >= 5) {
            if (auto status = tensor_to_int64(*inputs[4], steps);
                !status.has_value()) {
                return status;
            }
        } else if (has_steps_) {
            steps = steps_;
        }
    } else {
        if (!has_starts_ || !has_ends_) {
            return Error::invalid_argument(
                "Slice requires starts and ends attributes or inputs");
        }
        starts = starts_;
        ends = ends_;
        if (has_axes_) {
            axes = axes_;
        }
        if (has_steps_) {
            steps = steps_;
        }
    }

    if (starts.empty() || starts.size() != ends.size()) {
        return Error::invalid_argument(
            "Slice starts and ends must have the same non-zero length");
    }
    const int64_t count = static_cast<int64_t>(starts.size());
    if (count > rank) {
        return Error::invalid_argument(
            "Slice parameter count exceeds the input rank");
    }
    if (axes.empty()) {
        axes.resize(starts.size());
        for (std::size_t i = 0; i < axes.size(); ++i) {
            axes[i] = static_cast<int64_t>(i);
        }
    } else if (axes.size() != starts.size()) {
        return Error::invalid_argument(
            "Slice axes must match the starts length");
    }
    if (steps.empty()) {
        steps.assign(starts.size(), 1);
    } else if (steps.size() != starts.size()) {
        return Error::invalid_argument(
            "Slice steps must match the starts length");
    }

    std::vector<int64_t> out_dims = dims;
    std::vector<int64_t> start_by_axis(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> step_by_axis(static_cast<std::size_t>(rank), 1);
    std::vector<bool> sliced(static_cast<std::size_t>(rank), false);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        int64_t axis = 0;
        if (auto status = resolve_axis(axes[i], rank, axis);
            !status.has_value()) {
            return status;
        }
        if (sliced[static_cast<std::size_t>(axis)]) {
            return Error::invalid_argument(
                "Slice axes must be unique");
        }
        sliced[static_cast<std::size_t>(axis)] = true;

        const int64_t dim = dims[static_cast<std::size_t>(axis)];
        const int64_t step = steps[i];
        if (step == 0 ||
            step == std::numeric_limits<int64_t>::min()) {
            return Error::invalid_argument(
                "Slice step must be non-zero");
        }
        int64_t start = starts[i];
        int64_t end = ends[i];
        if (start < 0) {
            start += dim;
        }
        if (end < 0) {
            end += dim;
        }

        int64_t length = 0;
        if (step > 0) {
            start = std::clamp(start, int64_t{0}, dim);
            end = std::clamp(end, int64_t{0}, dim);
            if (start < end) {
                length = 1 + (end - start - 1) / step;
            }
        } else {
            start = std::clamp(start, int64_t{0}, dim - 1);
            end = std::clamp(end, int64_t{-1}, dim - 1);
            if (start > end) {
                const int64_t magnitude = -step;
                length = 1 + (start - end - 1) / magnitude;
            }
        }
        if (length <= 0) {
            return Error::invalid_argument(
                "Slice produces an empty tensor");
        }
        out_dims[static_cast<std::size_t>(axis)] = length;
        start_by_axis[static_cast<std::size_t>(axis)] = start;
        step_by_axis[static_cast<std::size_t>(axis)] = step;
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Slice requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t out_numel = out_shape.numel();
    for (int64_t out_index = 0; out_index < out_numel; ++out_index) {
        int64_t remainder = out_index;
        int64_t in_index = 0;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t stride = out_shape.stride(d);
            const int64_t coord = remainder / stride;
            remainder %= stride;
            const int64_t in_coord =
                sliced[static_cast<std::size_t>(d)]
                    ? start_by_axis[static_cast<std::size_t>(d)] +
                          coord * step_by_axis[static_cast<std::size_t>(d)]
                    : coord;
            in_index += in_coord * input.shape.stride(d);
        }
        out_data[out_index] = in_data[in_index];
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status PadOp::configure(const Attrs& attrs) {
    has_pads_ = false;
    has_value_ = false;
    pads_.clear();
    value_ = 0.0;

    detail::StringAttr mode;
    if (auto status = detail::read_string(attrs, "mode", mode);
        !status.has_value()) {
        return status;
    }
    if (mode.found && !mode.value.empty() && mode.value != "constant") {
        return Error::unsupported("Pad supports constant mode only");
    }

    detail::IntListAttr pads;
    if (auto status = detail::read_int_list(attrs, "pads", pads);
        !status.has_value()) {
        return status;
    }
    has_pads_ = pads.found;
    pads_ = std::move(pads.values);

    detail::DoubleAttr value;
    if (auto status = detail::read_double(attrs, "value", value);
        !status.has_value()) {
        return status;
    }
    has_value_ = value.found;
    value_ = value.value;
    return {};
}

Status PadOp::forward(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs,
                      ExecutionContext& context) {
    if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Pad expects one to four inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Pad requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (auto valid = require_positive_dims(dims, "Pad");
        !valid.has_value()) {
        return valid;
    }

    std::vector<int64_t> pads;
    if (inputs.size() >= 2) {
        if (auto status = tensor_to_int64(*inputs[1], pads);
            !status.has_value()) {
            return status;
        }
    } else if (has_pads_) {
        pads = pads_;
    } else {
        return Error::invalid_argument(
            "Pad requires a pads attribute or input");
    }

    std::vector<int64_t> axes;
    if (inputs.size() >= 4) {
        if (auto status = tensor_to_int64(*inputs[3], axes);
            !status.has_value()) {
            return status;
        }
    } else {
        axes.resize(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            axes[static_cast<std::size_t>(i)] = i;
        }
    }
    if (axes.empty() ||
        pads.size() != axes.size() * 2) {
        return Error::invalid_argument(
            "Pad pads must contain two values per axis");
    }

    float fill = static_cast<float>(value_);
    if (inputs.size() >= 3) {
        if (inputs[2]->shape.numel() != 1) {
            return Error::invalid_argument(
                "Pad constant value must be a scalar");
        }
        fill = inputs[2]->data<float>()[0];
    }

    std::vector<int64_t> out_dims = dims;
    std::vector<int64_t> begin_by_axis(static_cast<std::size_t>(rank), 0);
    std::vector<bool> padded(static_cast<std::size_t>(rank), false);
    for (std::size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = 0;
        if (auto status = resolve_axis(axes[i], rank, axis);
            !status.has_value()) {
            return status;
        }
        if (padded[static_cast<std::size_t>(axis)]) {
            return Error::invalid_argument(
                "Pad axes must be unique");
        }
        padded[static_cast<std::size_t>(axis)] = true;

        const int64_t begin = pads[i];
        const int64_t end = pads[i + axes.size()];
        int64_t out_dim = 0;
        if (!checked_add(dims[static_cast<std::size_t>(axis)], begin,
                         out_dim) ||
            !checked_add(out_dim, end, out_dim)) {
            return Error::invalid_argument("Pad output size overflows int64");
        }
        if (out_dim <= 0) {
            return Error::invalid_argument(
                "Pad crops the input to an empty dimension");
        }
        out_dims[static_cast<std::size_t>(axis)] = out_dim;
        begin_by_axis[static_cast<std::size_t>(axis)] = begin;
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Pad requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t out_numel = out_shape.numel();
    for (int64_t out_index = 0; out_index < out_numel; ++out_index) {
        int64_t remainder = out_index;
        int64_t in_index = 0;
        bool inside = true;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t stride = out_shape.stride(d);
            const int64_t coord = remainder / stride;
            remainder %= stride;
            const int64_t dim = dims[static_cast<std::size_t>(d)];
            const int64_t begin = begin_by_axis[static_cast<std::size_t>(d)];
            if (begin > coord || begin < coord - (dim - 1)) {
                inside = false;
                break;
            }
            in_index += (coord - begin) * input.shape.stride(d);
        }
        out_data[out_index] = inside ? in_data[in_index] : fill;
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status UpsampleOp::configure(const Attrs& attrs) {
    has_scales_ = false;
    scales_.clear();

    detail::StringAttr mode;
    if (auto status = detail::read_string(attrs, "mode", mode);
        !status.has_value()) {
        return status;
    }
    if (mode.found && !mode.value.empty() && mode.value != "nearest") {
        return Error::unsupported(
            "Upsample supports nearest mode only");
    }

    if (auto status = read_numeric_list(attrs, "scales", has_scales_, scales_);
        !status.has_value()) {
        return status;
    }
    return {};
}

Status UpsampleOp::forward(const std::vector<Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ExecutionContext& context) {
    if ((inputs.size() != 1 && inputs.size() != 2) ||
        outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Upsample expects one or two inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Upsample requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    if (dims.size() != 4) {
        return Error::invalid_argument(
            "Upsample supports NCHW rank-4 inputs only");
    }
    if (auto valid = require_positive_dims(dims, "Upsample");
        !valid.has_value()) {
        return valid;
    }

    std::vector<double> scales;
    if (inputs.size() == 2) {
        const Tensor& scale_tensor = *inputs[1];
        const int64_t count = scale_tensor.shape.numel();
        if (count != 4) {
            return Error::invalid_argument(
                "Upsample scales must contain four values");
        }
        const float* data = scale_tensor.data<float>();
        scales.assign(data, data + count);
    } else if (has_scales_) {
        scales = scales_;
    } else {
        return Error::invalid_argument(
            "Upsample requires a scales attribute or input");
    }
    if (scales.size() != dims.size()) {
        return Error::invalid_argument(
            "Upsample scales length must match the input rank");
    }

    std::vector<int64_t> out_dims(static_cast<std::size_t>(dims.size()));
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const double scale = scales[i];
        if (!std::isfinite(scale) || scale <= 0.0) {
            return Error::invalid_argument(
                "Upsample scales must be positive and finite");
        }
        const double scaled = static_cast<double>(dims[i]) * scale;
        if (!std::isfinite(scaled) || scaled > 9.2e18) {
            return Error::invalid_argument(
                "Upsample output size overflows int64");
        }
        const int64_t out_dim =
            static_cast<int64_t>(std::floor(scaled));
        if (out_dim <= 0) {
            return Error::invalid_argument(
                "Upsample produces an empty dimension");
        }
        out_dims[i] = out_dim;
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Upsample requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t out_numel = out_shape.numel();
    for (int64_t out_index = 0; out_index < out_numel; ++out_index) {
        int64_t remainder = out_index;
        int64_t in_index = 0;
        for (int64_t d = 0; d < 4; ++d) {
            const int64_t stride = out_shape.stride(d);
            const int64_t coord = remainder / stride;
            remainder %= stride;
            const int64_t dim = dims[static_cast<std::size_t>(d)];
            const double scale = scales[static_cast<std::size_t>(d)];
            int64_t in_coord = static_cast<int64_t>(
                std::floor(static_cast<double>(coord) / scale));
            if (in_coord < 0) {
                in_coord = 0;
            } else if (in_coord >= dim) {
                in_coord = dim - 1;
            }
            in_index += in_coord * input.shape.stride(d);
        }
        out_data[out_index] = in_data[in_index];
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status GatherOp::configure(const Attrs& attrs) {
    axis_ = 0;
    detail::IntAttr axis;
    if (auto status = detail::read_int(attrs, "axis", axis);
        !status.has_value()) {
        return status;
    }
    if (axis.found) {
        axis_ = axis.value;
    }
    return {};
}

Status GatherOp::forward(const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& outputs,
                         ExecutionContext& context) {
    if (inputs.size() != 2 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Gather expects two inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Gather requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& data = *inputs[0];
    const Tensor& index_tensor = *inputs[1];
    const std::vector<int64_t>& dims = data.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (auto valid = require_positive_dims(dims, "Gather");
        !valid.has_value()) {
        return valid;
    }
    if (index_tensor.shape.dims.empty() || index_tensor.shape.numel() <= 0) {
        return Error::invalid_argument(
            "Gather indices must be a non-empty tensor");
    }

    int64_t axis = 0;
    if (auto status = resolve_axis(axis_, rank, axis); !status.has_value()) {
        return status;
    }
    const int64_t axis_dim = dims[static_cast<std::size_t>(axis)];

    std::vector<int64_t> indices;
    if (auto status = tensor_to_int64(index_tensor, indices);
        !status.has_value()) {
        return status;
    }
    for (int64_t& index : indices) {
        if (index < 0) {
            index += axis_dim;
        }
        if (index < 0 || index >= axis_dim) {
            return Error::invalid_argument(
                "Gather index is out of range");
        }
    }

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<std::size_t>(rank) +
                     index_tensor.shape.dims.size());
    for (int64_t d = 0; d < axis; ++d) {
        out_dims.push_back(dims[static_cast<std::size_t>(d)]);
    }
    for (int64_t dim : index_tensor.shape.dims) {
        out_dims.push_back(dim);
    }
    for (int64_t d = axis + 1; d < rank; ++d) {
        out_dims.push_back(dims[static_cast<std::size_t>(d)]);
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Gather requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    int64_t outer = 1;
    int64_t inner = 1;
    if (auto status = checked_product(dims, 0,
                                      static_cast<std::size_t>(axis), outer);
        !status.has_value()) {
        return status;
    }
    if (auto status = checked_product(dims,
                                      static_cast<std::size_t>(axis) + 1,
                                      static_cast<std::size_t>(rank), inner);
        !status.has_value()) {
        return status;
    }

    const float* in_data = data.data<float>();
    float* out_data = result->data<float>();
    const int64_t index_count = static_cast<int64_t>(indices.size());
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < index_count; ++j) {
            const int64_t source =
                (o * axis_dim + indices[static_cast<std::size_t>(j)]) * inner;
            const int64_t destination = (o * index_count + j) * inner;
            std::copy(in_data + source, in_data + source + inner,
                      out_data + destination);
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status ReduceOp::configure(const Attrs& attrs) {
    has_axes_ = false;
    axes_.clear();
    keepdims_ = true;
    mode_ = Mode::Mean;

    detail::IntListAttr axes;
    if (auto status = detail::read_int_list(attrs, "axes", axes);
        !status.has_value()) {
        return status;
    }
    has_axes_ = axes.found;
    axes_ = std::move(axes.values);

    detail::IntAttr keepdims;
    if (auto status = detail::read_int(attrs, "keepdims", keepdims);
        !status.has_value()) {
        return status;
    }
    if (keepdims.found) {
        if (keepdims.value != 0 && keepdims.value != 1) {
            return Error::invalid_argument(
                "Reduce keepdims must be 0 or 1");
        }
        keepdims_ = keepdims.value != 0;
    }

    detail::StringAttr mode;
    if (auto status = detail::read_string(attrs, "reduce_mode", mode);
        !status.has_value()) {
        return status;
    }
    if (mode.found) {
        if (mode.value == "mean") {
            mode_ = Mode::Mean;
        } else if (mode.value == "sum") {
            mode_ = Mode::Sum;
        } else {
            return Error::invalid_argument(
                "Reduce reduce_mode must be 'mean' or 'sum'");
        }
    }
    return {};
}

Status ReduceOp::forward(const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& outputs,
                         ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Reduce expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Reduce requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (auto valid = require_positive_dims(dims, "Reduce");
        !valid.has_value()) {
        return valid;
    }

    std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
    if (has_axes_ && !axes_.empty()) {
        if (static_cast<int64_t>(axes_.size()) > rank) {
            return Error::invalid_argument(
                "Reduce axes count exceeds the input rank");
        }
        for (int64_t raw_axis : axes_) {
            int64_t axis = 0;
            if (auto status = resolve_axis(raw_axis, rank, axis);
                !status.has_value()) {
                return status;
            }
            if (reduced[static_cast<std::size_t>(axis)]) {
                return Error::invalid_argument(
                    "Reduce axes must be unique");
            }
            reduced[static_cast<std::size_t>(axis)] = true;
        }
    } else {
        reduced.assign(static_cast<std::size_t>(rank), true);
    }

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<std::size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        if (reduced[static_cast<std::size_t>(d)]) {
            if (keepdims_) {
                out_dims.push_back(1);
            }
        } else {
            out_dims.push_back(dims[static_cast<std::size_t>(d)]);
        }
    }
    // A fully reduced keepdims=0 result is a scalar in ONNX. The engine
    // represents scalars as a single-element tensor.
    if (out_dims.empty()) {
        out_dims.push_back(1);
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Reduce requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const int64_t out_numel = out_shape.numel();
    float* out_data = result->data<float>();
    std::fill(out_data, out_data + out_numel, 0.0f);

    int64_t reduce_count = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (reduced[static_cast<std::size_t>(d)]) {
            if (!checked_mul(reduce_count, dims[static_cast<std::size_t>(d)],
                             reduce_count)) {
                return Error::invalid_argument(
                    "Reduce element count overflows int64");
            }
        }
    }

    const float* in_data = input.data<float>();
    const int64_t in_numel = input.shape.numel();
    std::vector<int64_t> coord(static_cast<std::size_t>(rank), 0);
    for (int64_t in_index = 0; in_index < in_numel; ++in_index) {
        int64_t remainder = in_index;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t stride = input.shape.stride(d);
            coord[static_cast<std::size_t>(d)] = remainder / stride;
            remainder %= stride;
        }
        int64_t out_index = 0;
        int64_t factor = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (reduced[static_cast<std::size_t>(d)]) {
                continue;
            }
            out_index +=
                coord[static_cast<std::size_t>(d)] * factor;
            factor *= dims[static_cast<std::size_t>(d)];
        }
        out_data[out_index] += in_data[in_index];
    }

    if (mode_ == Mode::Mean) {
        const float divisor = static_cast<float>(reduce_count);
        for (int64_t i = 0; i < out_numel; ++i) {
            out_data[i] /= divisor;
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

Status ClipOp::configure(const Attrs& attrs) {
    has_min_ = false;
    has_max_ = false;
    min_ = 0.0;
    max_ = 0.0;

    detail::DoubleAttr min;
    if (auto status = detail::read_double(attrs, "min", min);
        !status.has_value()) {
        return status;
    }
    has_min_ = min.found;
    min_ = min.value;

    detail::DoubleAttr max;
    if (auto status = detail::read_double(attrs, "max", max);
        !status.has_value()) {
        return status;
    }
    has_max_ = max.found;
    max_ = max.value;
    return {};
}

Status ClipOp::forward(const std::vector<Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ExecutionContext& context) {
    if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Clip expects one to three inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Clip requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    float lower = -std::numeric_limits<float>::infinity();
    float upper = std::numeric_limits<float>::infinity();
    if (inputs.size() >= 2) {
        if (inputs[1]->shape.numel() != 1) {
            return Error::invalid_argument(
                "Clip min must be a scalar");
        }
        lower = inputs[1]->data<float>()[0];
    } else if (has_min_) {
        lower = static_cast<float>(min_);
    }
    if (inputs.size() >= 3) {
        if (inputs[2]->shape.numel() != 1) {
            return Error::invalid_argument(
                "Clip max must be a scalar");
        }
        upper = inputs[2]->data<float>()[0];
    } else if (has_max_) {
        upper = static_cast<float>(max_);
    }
    if (std::isnan(lower) || std::isnan(upper) || lower > upper) {
        return Error::invalid_argument(
            "Clip bounds must be ordered and not NaN");
    }

    const Tensor& input = *inputs[0];
    if (!input.shape.is_concrete()) {
        return Error::invalid_argument(
            "Clip requires a concrete input shape");
    }
    auto result = Tensor::allocate(input.shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t count = input.shape.numel();
    for (int64_t i = 0; i < count; ++i) {
        const float value = in_data[i];
        out_data[i] = value < lower ? lower : (value > upper ? upper : value);
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace ie
