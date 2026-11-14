#include "ops/traditional/layout_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

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

Status checked_product(const std::vector<int64_t>& dims, int64_t begin,
                       int64_t end, int64_t& out) {
    int64_t product = 1;
    for (int64_t i = begin; i < end; ++i) {
        if (!checked_mul(product, dims[static_cast<std::size_t>(i)],
                         product)) {
            return Error::invalid_argument("shape product overflows int64");
        }
    }
    out = product;
    return {};
}

bool shape_value_to_int(float value, int64_t& out) {
    // Shape values are normally small non-negative integers. Reject values that
    // cannot safely be represented as int64 before casting.
    if (!std::isfinite(value) || std::fabs(value) > 1e18f) {
        return false;
    }
    out = static_cast<int64_t>(value);
    return true;
}

}  // namespace

Status FlattenOp::configure(const Attrs& attrs) {
    axis_ = 1;
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

Status FlattenOp::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Flatten expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Flatten requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (rank < 1) {
        return Error::invalid_argument(
            "Flatten does not support scalar inputs");
    }

    int64_t axis = axis_;
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis > rank) {
        return Error::invalid_argument("Flatten axis is out of range");
    }

    int64_t before = 1;
    int64_t after = 1;
    if (auto status = checked_product(dims, 0, axis, before);
        !status.has_value()) {
        return status;
    }
    if (auto status = checked_product(dims, axis, rank, after);
        !status.has_value()) {
        return status;
    }
    if (before <= 0 || after <= 0) {
        return Error::invalid_argument(
            "Flatten requires concrete input dimensions");
    }

    auto result = Tensor::allocate(Shape{{before, after}}, DType::F32,
                                   context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t count = input.shape.numel();
    std::copy(in_data, in_data + count, out_data);

    *outputs[0] = std::move(*result);
    return {};
}

Status ReshapeOp::configure(const Attrs& attrs) {
    has_shape_attr_ = false;
    shape_attr_.clear();

    detail::IntListAttr shape;
    if (auto status = detail::read_int_list(attrs, "shape", shape);
        !status.has_value()) {
        return status;
    }
    if (shape.found) {
        has_shape_attr_ = true;
        shape_attr_ = std::move(shape.values);
    }
    return {};
}

Status ReshapeOp::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          ExecutionContext& context) {
    if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Reshape expects one or two inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Reshape requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t input_numel = input.shape.numel();
    if (input_numel <= 0) {
        return Error::invalid_argument(
            "Reshape requires a non-empty concrete input");
    }

    std::vector<int64_t> target;
    if (inputs.size() == 2) {
        const Tensor& shape_tensor = *inputs[1];
        const int64_t shape_count = shape_tensor.shape.numel();
        if (shape_count <= 0) {
            return Error::invalid_argument(
                "Reshape target shape must not be empty");
        }
        const float* shape_data = shape_tensor.data<float>();
        target.resize(static_cast<std::size_t>(shape_count));
        for (int64_t i = 0; i < shape_count; ++i) {
            if (!shape_value_to_int(shape_data[i],
                                    target[static_cast<std::size_t>(i)])) {
                return Error::invalid_argument(
                    "Reshape target shape contains an invalid value");
            }
        }
    } else if (has_shape_attr_) {
        target = shape_attr_;
    } else {
        return Error::invalid_argument(
            "Reshape requires a shape input or shape attribute");
    }

    const int64_t target_rank = static_cast<int64_t>(target.size());
    if (target_rank < 1) {
        return Error::invalid_argument(
            "Reshape target shape must not be empty");
    }

    std::vector<int64_t> out_dims(static_cast<std::size_t>(target_rank), 1);
    int64_t inferred_index = -1;
    for (int64_t i = 0; i < target_rank; ++i) {
        const int64_t value = target[static_cast<std::size_t>(i)];
        if (value == -1) {
            if (inferred_index >= 0) {
                return Error::invalid_argument(
                    "Reshape allows at most one inferred dimension");
            }
            inferred_index = i;
        } else if (value == 0) {
            // allowzero=0 semantics: copy the matching input dimension.
            if (i >= rank) {
                return Error::invalid_argument(
                    "Reshape cannot copy a missing input dimension");
            }
            out_dims[static_cast<std::size_t>(i)] =
                dims[static_cast<std::size_t>(i)];
            if (out_dims[static_cast<std::size_t>(i)] <= 0) {
                return Error::invalid_argument(
                    "Reshape copied a non-concrete input dimension");
            }
        } else if (value > 0) {
            out_dims[static_cast<std::size_t>(i)] = value;
        } else {
            return Error::invalid_argument(
                "Reshape target dimensions must be 0, -1, or positive");
        }
    }

    int64_t known = 1;
    for (int64_t i = 0; i < target_rank; ++i) {
        if (i == inferred_index) {
            continue;
        }
        if (!checked_mul(known, out_dims[static_cast<std::size_t>(i)],
                         known)) {
            return Error::invalid_argument("Reshape target shape overflows");
        }
    }

    if (inferred_index >= 0) {
        if (known <= 0 || input_numel % known != 0) {
            return Error::invalid_argument(
                "Reshape element counts do not match");
        }
        const int64_t inferred = input_numel / known;
        if (inferred <= 0) {
            return Error::invalid_argument(
                "Reshape inferred dimension is not positive");
        }
        out_dims[static_cast<std::size_t>(inferred_index)] = inferred;
    } else if (known != input_numel) {
        return Error::invalid_argument("Reshape element counts do not match");
    }

    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Reshape requires a concrete output shape");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    std::copy(in_data, in_data + input_numel, out_data);

    *outputs[0] = std::move(*result);
    return {};
}

Status TransposeOp::configure(const Attrs& attrs) {
    has_perm_ = false;
    perm_.clear();

    detail::IntListAttr perm;
    if (auto status = detail::read_int_list(attrs, "perm", perm);
        !status.has_value()) {
        return status;
    }
    if (perm.found) {
        has_perm_ = true;
        perm_ = std::move(perm.values);
    }
    return {};
}

Status TransposeOp::forward(const std::vector<Tensor*>& inputs,
                            std::vector<Tensor*>& outputs,
                            ExecutionContext& context) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Transpose expects one input and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Transpose requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& input = *inputs[0];
    const std::vector<int64_t>& dims = input.shape.dims;
    const int64_t rank = static_cast<int64_t>(dims.size());
    if (rank < 1) {
        return Error::invalid_argument(
            "Transpose does not support scalar inputs");
    }

    std::vector<int64_t> perm;
    if (has_perm_) {
        perm = perm_;
    } else {
        perm.resize(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            perm[static_cast<std::size_t>(i)] = rank - 1 - i;
        }
    }
    if (static_cast<int64_t>(perm.size()) != rank) {
        return Error::invalid_argument(
            "Transpose perm length must match the input rank");
    }

    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t axis = perm[static_cast<std::size_t>(i)];
        if (axis < 0 || axis >= rank ||
            seen[static_cast<std::size_t>(axis)]) {
            return Error::invalid_argument(
                "Transpose perm is not a valid permutation");
        }
        seen[static_cast<std::size_t>(axis)] = true;
    }

    std::vector<int64_t> out_dims(static_cast<std::size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        out_dims[static_cast<std::size_t>(i)] =
            dims[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])];
    }
    const Shape out_shape{std::move(out_dims)};
    if (!out_shape.is_concrete()) {
        return Error::invalid_argument(
            "Transpose requires concrete input dimensions");
    }

    auto result = Tensor::allocate(out_shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* in_data = input.data<float>();
    float* out_data = result->data<float>();
    const int64_t count = input.shape.numel();
    for (int64_t out_index = 0; out_index < count; ++out_index) {
        int64_t remainder = out_index;
        int64_t in_index = 0;
        for (int64_t i = 0; i < rank; ++i) {
            const int64_t out_stride = out_shape.stride(i);
            const int64_t coord = remainder / out_stride;
            remainder %= out_stride;
            in_index += coord * input.shape.stride(
                                   perm[static_cast<std::size_t>(i)]);
        }
        out_data[out_index] = in_data[in_index];
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace ie
