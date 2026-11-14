#include "ops/traditional/gemm_op.h"

#include <cstddef>
#include <vector>

namespace ie {
namespace {

// Local reference row kernel used when the execution context has no injected
// matmul_row implementation. Accumulation order is ascending k.
void gemm_row_reference(const float* a_row, const float* b, float* out_row,
                        std::int64_t k, std::int64_t n) {
    for (std::int64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (std::int64_t t = 0; t < k; ++t) {
            acc += a_row[t] * b[t * n + j];
        }
        out_row[j] = acc;
    }
}

enum class CBroadcast { None, Scalar, Row, Column, Full };

}  // namespace

Status GemmOp::configure(const Attrs& attrs) {
    alpha_ = 1.0;
    beta_ = 1.0;
    trans_a_ = 0;
    trans_b_ = 0;

    detail::DoubleAttr alpha;
    if (auto status = detail::read_double(attrs, "alpha", alpha);
        !status.has_value()) {
        return status;
    }
    if (alpha.found) {
        alpha_ = alpha.value;
    }

    detail::DoubleAttr beta;
    if (auto status = detail::read_double(attrs, "beta", beta);
        !status.has_value()) {
        return status;
    }
    if (beta.found) {
        beta_ = beta.value;
    }

    detail::IntAttr trans_a;
    if (auto status = detail::read_int(attrs, "transA", trans_a);
        !status.has_value()) {
        return status;
    }
    if (trans_a.found) {
        if (trans_a.value != 0 && trans_a.value != 1) {
            return Error::invalid_argument("Gemm transA must be 0 or 1");
        }
        trans_a_ = trans_a.value;
    }

    detail::IntAttr trans_b;
    if (auto status = detail::read_int(attrs, "transB", trans_b);
        !status.has_value()) {
        return status;
    }
    if (trans_b.found) {
        if (trans_b.value != 0 && trans_b.value != 1) {
            return Error::invalid_argument("Gemm transB must be 0 or 1");
        }
        trans_b_ = trans_b.value;
    }

    return {};
}

Status GemmOp::forward(const std::vector<Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ExecutionContext& context) {
    if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
        outputs[0] == nullptr) {
        return Error::invalid_argument(
            "Gemm expects two or three inputs and one output");
    }
    if (context.allocator == nullptr) {
        return Error::invalid_argument("Gemm requires an allocator");
    }
    if (auto valid = validate_p0_kernel_inputs(inputs); !valid.has_value()) {
        return valid.error();
    }

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];
    if (a.shape.dims.size() != 2 || b.shape.dims.size() != 2) {
        return Error::invalid_argument("Gemm expects 2D A and B tensors");
    }

    const bool trans_a = trans_a_ != 0;
    const bool trans_b = trans_b_ != 0;
    const std::int64_t m = trans_a ? a.shape.dims[1] : a.shape.dims[0];
    const std::int64_t k_a = trans_a ? a.shape.dims[0] : a.shape.dims[1];
    const std::int64_t k_b = trans_b ? b.shape.dims[1] : b.shape.dims[0];
    const std::int64_t n = trans_b ? b.shape.dims[0] : b.shape.dims[1];
    if (m <= 0 || k_a <= 0 || n <= 0 || k_b <= 0) {
        return Error::invalid_argument("Gemm expects positive dimensions");
    }
    if (k_a != k_b) {
        return Error::invalid_argument("Gemm inner dimensions mismatch");
    }
    const std::int64_t k = k_a;

    CBroadcast c_mode = CBroadcast::None;
    const float* c_data = nullptr;
    if (inputs.size() == 3) {
        const Tensor& c = *inputs[2];
        c_data = c.data<float>();
        const auto& dims = c.shape.dims;
        if (dims.empty() || c.shape.numel() == 1) {
            c_mode = CBroadcast::Scalar;
        } else if (dims.size() == 1) {
            // A rank-1 bias aligns with N (ONNX trailing-dim broadcast) and is
            // only treated as per-row when it cannot match N.
            if (dims[0] == n) {
                c_mode = CBroadcast::Column;
            } else if (dims[0] == m) {
                c_mode = CBroadcast::Row;
            } else {
                return Error::invalid_argument(
                    "Gemm bias must broadcast to [M,N]");
            }
        } else if (dims.size() == 2) {
            if (dims[0] == m && dims[1] == n) {
                c_mode = CBroadcast::Full;
            } else if (dims[0] == m && dims[1] == 1) {
                c_mode = CBroadcast::Row;
            } else if (dims[0] == 1 && dims[1] == n) {
                c_mode = CBroadcast::Column;
            } else {
                return Error::invalid_argument(
                    "Gemm bias must broadcast to [M,N]");
            }
        } else {
            return Error::invalid_argument(
                "Gemm bias must broadcast to [M,N]");
        }
    }

    auto result = Tensor::allocate(Shape{{m, n}}, DType::F32,
                                   context.allocator);
    if (!result.has_value()) {
        return result.error();
    }

    const auto c_at = [&](std::int64_t i, std::int64_t j) -> float {
        switch (c_mode) {
            case CBroadcast::None:
                return 0.0f;
            case CBroadcast::Scalar:
                return c_data[0];
            case CBroadcast::Row:
                return c_data[i];
            case CBroadcast::Column:
                return c_data[j];
            case CBroadcast::Full:
                return c_data[i * n + j];
        }
        return 0.0f;
    };

    const float alpha = static_cast<float>(alpha_);
    const float beta = static_cast<float>(beta_);
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* out_data = result->data<float>();

    const KernelFns::MatMulRowFn row_fn =
        (context.kernels != nullptr && context.kernels->matmul_row != nullptr)
            ? context.kernels->matmul_row
            : gemm_row_reference;

    std::vector<float> transposed_row;
    if (trans_a) {
        transposed_row.resize(static_cast<std::size_t>(k));
    }

    for (std::int64_t i = 0; i < m; ++i) {
        const float* a_row = nullptr;
        if (trans_a) {
            for (std::int64_t t = 0; t < k; ++t) {
                transposed_row[static_cast<std::size_t>(t)] =
                    a_data[t * m + i];
            }
            a_row = transposed_row.data();
        } else {
            a_row = a_data + i * k;
        }

        float* out_row = out_data + i * n;
        if (!trans_b) {
            row_fn(a_row, b_data, out_row, k, n);
            for (std::int64_t j = 0; j < n; ++j) {
                out_row[j] = alpha * out_row[j] +
                             (c_mode == CBroadcast::None
                                  ? 0.0f
                                  : beta * c_at(i, j));
            }
        } else {
            for (std::int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (std::int64_t t = 0; t < k; ++t) {
                    acc += a_row[t] * b_data[j * k + t];
                }
                out_row[j] = alpha * acc +
                             (c_mode == CBroadcast::None
                                  ? 0.0f
                                  : beta * c_at(i, j));
            }
        }
    }

    *outputs[0] = std::move(*result);
    return {};
}

}  // namespace ie
