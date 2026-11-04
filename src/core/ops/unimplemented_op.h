#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/ops/op.h"

namespace ie {

// Base for P0.5 interface-only ops. forward() never touches inputs/outputs and
// always reports Unsupported with the concrete op name.
class UnimplementedOp : public Op {
public:
    explicit UnimplementedOp(std::string name) : name_(std::move(name)) {}

    Status forward(const std::vector<Tensor*>& /*inputs*/,
                   std::vector<Tensor*>& /*outputs*/,
                   ExecutionContext& /*context*/) override {
        return Error::unsupported(name_ + " not implemented in P0.5");
    }

protected:
    std::string name_;
};

}  // namespace ie
