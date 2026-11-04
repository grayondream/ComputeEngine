#pragma once

#include <vector>

#include "core/ops/op.h"

namespace ie {

// Rule inference ops (docs/readme.md §8.1). P0.5 freezes the surface only;
// forward() reports Unsupported.
class RuleMatchOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& /*inputs*/,
                   std::vector<Tensor*>& /*outputs*/,
                   ExecutionContext& /*context*/) override {
        return Error::unsupported("rule match not implemented in P0.5");
    }
};

class RuleFireOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& /*inputs*/,
                   std::vector<Tensor*>& /*outputs*/,
                   ExecutionContext& /*context*/) override {
        return Error::unsupported("rule fire not implemented in P0.5");
    }
};

class FactAssertOp final : public Op {
public:
    Status forward(const std::vector<Tensor*>& /*inputs*/,
                   std::vector<Tensor*>& /*outputs*/,
                   ExecutionContext& /*context*/) override {
        return Error::unsupported("fact assert not implemented in P0.5");
    }
};

}  // namespace ie
