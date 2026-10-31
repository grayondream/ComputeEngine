#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/allocator.h"
#include "core/memory/tensor.h"
#include "core/ops/op_registry.h"

namespace ie {

class Executor {
public:
    explicit Executor(OpRegistry& registry = OpRegistry::instance())
        : registry_(registry) {
        register_builtin_ops(registry_);
    }

    Result<std::vector<Tensor>> run(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& inputs,
        Allocator& allocator);

private:
    OpRegistry& registry_;
};

}  // namespace ie
