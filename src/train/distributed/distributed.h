#pragma once

#include <string>
#include <string_view>

#include "core/base/error.h"
#include "core/memory/tensor.h"

namespace ie {

// Distributed training configuration. Single-process by default (P0.5).
struct DistributedConfig {
    int world_size{1};
    int rank{0};
    std::string backend;
};

class DistributedContext {
public:
    Status init(const DistributedConfig& config) {
        (void)config;
        return Error::unsupported("distributed training not implemented in P0.5");
    }

    Status barrier() {
        return Error::unsupported("distributed barrier not implemented in P0.5");
    }

    Status all_reduce(Tensor& tensor, std::string_view op) {
        (void)tensor;
        (void)op;
        return Error::unsupported(
            "distributed all_reduce not implemented in P0.5");
    }

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

private:
    int rank_{0};
    int world_size_{1};
};

}  // namespace ie
