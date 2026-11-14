#include "ops/traditional/batch_norm_op.h"

#include <cmath>
#include <cstddef>
#include <utility>

#include "ops/traditional/attrs_util.h"

namespace ie {

Status BatchNormOp::configure(const Attrs& attrs) {
    epsilon_ = 1e-5;
    detail::DoubleAttr epsilon;
    if (auto status = detail::read_double(attrs, "epsilon", epsilon);
        !status.has_value()) {
        return status;
    }
    if (epsilon.found) {
        if (epsilon.value < 0.0) {
            return Error::invalid_argument("epsilon must be non-negative");
        }
        epsilon_ = epsilon.value;
    }
    return {};
}

Status BatchNormOp::forward(const std::vector<Tensor*>& inputs,
                            std::vector<Tensor*>& outputs,
                            ExecutionContext& context) {
    if (inputs.size() != 5 || outputs.size() != 1 || outputs[0] == nullptr) {
        return Error::invalid_argument(
            "BatchNorm expects five inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("BatchNorm requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& x = *inputs[0];
    if (x.shape.dims.size() < 2) {
        return Error::invalid_argument(
            "BatchNorm expects at least a 2D input");
    }
    const std::int64_t n = x.shape.dims[0];
    const std::int64_t c = x.shape.dims[1];
    if (n <= 0 || c <= 0) {
        return Error::invalid_argument("BatchNorm expects positive dimensions");
    }
    std::int64_t spatial = 1;
    for (std::size_t i = 2; i < x.shape.dims.size(); ++i) {
        if (x.shape.dims[i] <= 0) {
            return Error::invalid_argument(
                "BatchNorm expects positive dimensions");
        }
        spatial *= x.shape.dims[i];
    }
    for (std::size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i]->numel() != c) {
            return Error::invalid_argument(
                "BatchNorm parameters must have C elements");
        }
    }

    auto result = Tensor::allocate(x.shape, DType::F32, context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const float* scale = inputs[1]->data<float>();
    const float* bias = inputs[2]->data<float>();
    const float* mean = inputs[3]->data<float>();
    const float* var = inputs[4]->data<float>();
    const float* in = x.data<float>();
    float* out = result->data<float>();

    for (std::int64_t channel = 0; channel < c; ++channel) {
        const double shifted = static_cast<double>(var[channel]) + epsilon_;
        if (!(shifted > 0.0)) {
            return Error::invalid_argument(
                "BatchNorm variance plus epsilon must be positive");
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(shifted));
        const float gain = scale[channel] * inv;
        const float shift = bias[channel] - mean[channel] * gain;
        const std::int64_t begin = channel * spatial;
        for (std::int64_t batch = 0; batch < n; ++batch) {
            const std::int64_t offset = batch * c * spatial + begin;
            for (std::int64_t i = 0; i < spatial; ++i) {
                out[offset + i] =
                    in[offset + i] * gain + shift;
            }
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace ie
