#pragma once

#include <string>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/tensor.h"

namespace ie {

// Developer debug helpers (graph/tensor dumps, profiling). Profiling lifecycle
// hooks are no-ops; dump/snapshot operations are not implemented in P0.5.
class DebugTools {
public:
    static Status dump_graph(const Graph& graph, const std::string& path) {
        (void)graph;
        (void)path;
        return Error::unsupported(
            "graph dump not implemented in P0.5");
    }

    static Status dump_tensor(const Tensor& tensor, const std::string& path) {
        (void)tensor;
        (void)path;
        return Error::unsupported(
            "tensor dump not implemented in P0.5");
    }

    static Status dump_memory(const std::string& path) {
        (void)path;
        return Error::unsupported(
            "memory dump not implemented in P0.5");
    }

    static Status validate_graph(const Graph& graph) {
        (void)graph;
        return Error::unsupported(
            "graph validation tooling not implemented in P0.5");
    }

    static Status start_profiling() { return {}; }
    static Status stop_profiling() { return {}; }
};

}  // namespace ie
