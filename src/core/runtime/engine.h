#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/backend/backend.h"
#include "core/backend/engine_config.h"
#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/malloc_allocator.h"
#include "core/memory/tensor.h"

namespace ie {

class Engine {
public:
    explicit Engine(EngineConfig config);
    ~Engine();

    Status load(const Graph& graph);
    Result<std::vector<Tensor>> infer(
        const std::unordered_map<std::string, Tensor>& inputs);

private:
    EngineConfig config_;
    std::optional<Error> error_;
    std::optional<Graph> graph_;
    std::shared_ptr<MallocAllocator> allocator_;
    std::unique_ptr<Backend> backend_;
    bool loaded_{false};
};

}  // namespace ie
