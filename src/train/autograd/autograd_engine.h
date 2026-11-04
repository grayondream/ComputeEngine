#pragma once

#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "core/ops/op.h"

namespace ie {

// Autograd contract (docs/readme.md §9.2). P0.5 freezes the surface only;
// graph recording and gradient computation arrive with the training milestone.
class AutogradEngine {
public:
    // Recording is a no-op until the autograd milestone.
    void record(const Op& op, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs) {
        (void)op;
        (void)inputs;
        (void)outputs;
    }

    Status backward(Tensor& loss) {
        (void)loss;
        return Error::unsupported("autograd backward not implemented in P0.5");
    }

    Result<Tensor*> grad(const Tensor& tensor) {
        (void)tensor;
        return Error::unsupported("autograd grad not implemented in P0.5");
    }

    void zero_grad() {}

    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }

private:
    bool training_{false};
};

}  // namespace ie
