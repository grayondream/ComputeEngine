#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/ir/pass.h"

namespace ie {

// Runs passes in registration order and stops at the first failure.
class PassManager {
public:
    void add(std::unique_ptr<Pass> pass) {
        if (pass) {
            passes_.push_back(std::move(pass));
        }
    }

    Status run(Graph& graph, PassContext& context) {
        for (const auto& pass : passes_) {
            Status status = pass->run(graph, context);
            if (!status) {
                return status;
            }
        }
        return {};
    }

    std::size_t size() const { return passes_.size(); }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace ie
