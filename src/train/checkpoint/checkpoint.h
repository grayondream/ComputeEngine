#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/allocator.h"
#include "core/memory/tensor.h"

namespace ie {

// Checkpoint layout (docs/readme.md §9.4). The optional sections are filled in
// by the training milestone; P0.5 only freezes the structure.
struct Checkpoint {
    std::unordered_map<std::string, Tensor> model_weights;

    std::optional<std::unordered_map<std::string, Tensor>> optimizer_state;
    std::optional<std::unordered_map<std::string, Tensor>> scheduler_state;
    std::optional<std::unordered_map<std::string, BufferHandle>> rng_state;
    std::optional<int> global_step;
    std::optional<int> epoch;
    std::optional<Graph> graph;
};

class CheckpointIO {
public:
    static Status save(const Checkpoint& checkpoint, const std::string& path) {
        (void)checkpoint;
        (void)path;
        return Error::unsupported("checkpoint save not implemented in P0.5");
    }

    static Result<Checkpoint> load(const std::string& path) {
        (void)path;
        return Error::unsupported("checkpoint load not implemented in P0.5");
    }

    static Result<std::unordered_map<std::string, Tensor>> load_weights_only(
        const std::string& path) {
        (void)path;
        return Error::unsupported(
            "checkpoint weight loading not implemented in P0.5");
    }
};

}  // namespace ie
