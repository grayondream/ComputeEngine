#pragma once

#include <vector>

#include "core/base/error.h"
#include "core/ir/attr.h"
#include "core/ir/graph.h"
#include "core/memory/tensor.h"
#include "core/runtime/execution_context.h"

namespace ie {

// Shared guard for the P0 reference kernels: no null tensors/buffers, F32 only.
inline Status validate_p0_kernel_inputs(const std::vector<Tensor*>& inputs) {
    for (Tensor* tensor : inputs) {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            return Error::invalid_argument("null tensor or buffer");
        }
        if (tensor->dtype != DType::F32) {
            return Error::unsupported("P0 kernels support F32 only");
        }
    }
    return {};
}

// Op interface (docs/readme.md §3.6). P0 implements forward only.
class Op {
public:
    virtual ~Op() = default;

    // Optional per-node configuration (e.g. scalar attrs) applied by the
    // executor before forward().
    virtual Status configure(const Attrs& /*attrs*/) { return {}; }

    virtual Status forward(const std::vector<Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ExecutionContext& context) = 0;

    virtual Status backward(const std::vector<Tensor*>& /*inputs*/,
                            const std::vector<Tensor*>& /*outputs*/,
                            const std::vector<Tensor*>& /*grad_outputs*/,
                            std::vector<Tensor*>& /*grad_inputs*/,
                            ExecutionContext& /*context*/) {
        return Error::unsupported("backward not implemented");
    }

    virtual bool supports_training() const { return false; }

    virtual bool is_control_flow() const { return false; }
    virtual Status execute_subgraph(ExecutionContext& /*context*/,
                                    const Subgraph& /*subgraph*/) {
        return Error::unsupported("subgraph not implemented");
    }
};

}  // namespace ie
