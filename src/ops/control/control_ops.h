#pragma once

#include "core/ops/unimplemented_op.h"

namespace ie {

// Control-flow ops (docs/readme.md §3.6). P0.5 freezes the surface only; the
// executor cannot run subgraphs yet.

class IfOp final : public UnimplementedOp {
public:
    IfOp() : UnimplementedOp("If") {}

    bool is_control_flow() const override { return true; }

    Status execute_subgraph(ExecutionContext& /*context*/,
                            const Subgraph& /*subgraph*/) override {
        return Error::unsupported(name_ + " subgraph not implemented in P0.5");
    }
};

class LoopOp final : public UnimplementedOp {
public:
    LoopOp() : UnimplementedOp("Loop") {}

    bool is_control_flow() const override { return true; }

    Status execute_subgraph(ExecutionContext& /*context*/,
                            const Subgraph& /*subgraph*/) override {
        return Error::unsupported(name_ + " subgraph not implemented in P0.5");
    }
};

class WhileOp final : public UnimplementedOp {
public:
    WhileOp() : UnimplementedOp("While") {}

    bool is_control_flow() const override { return true; }

    Status execute_subgraph(ExecutionContext& /*context*/,
                            const Subgraph& /*subgraph*/) override {
        return Error::unsupported(name_ + " subgraph not implemented in P0.5");
    }
};

class SubgraphOp final : public UnimplementedOp {
public:
    SubgraphOp() : UnimplementedOp("Subgraph") {}

    bool is_control_flow() const override { return true; }

    Status execute_subgraph(ExecutionContext& /*context*/,
                            const Subgraph& /*subgraph*/) override {
        return Error::unsupported(name_ + " subgraph not implemented in P0.5");
    }
};

}  // namespace ie
