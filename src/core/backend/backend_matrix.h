#pragma once

#include <array>
#include <unordered_map>

#include "core/backend/backend_caps.h"
#include "core/base/types.h"
#include "core/ir/op_type.h"

namespace ie {

// Backend capability matrix (docs/readme.md §5.5). Index order matches
// BackendKind: CPU=0, CUDA=1, VULKAN=2.
struct BackendMatrix {
    std::unordered_map<OpType, std::array<bool, 3>> op_support;
    std::unordered_map<DType, std::array<bool, 3>> dtype_support;

    bool supports(BackendKind kind, OpType op) const {
        auto it = op_support.find(op);
        if (it == op_support.end()) {
            return false;
        }
        return it->second[static_cast<std::size_t>(kind)];
    }
};

}  // namespace ie
