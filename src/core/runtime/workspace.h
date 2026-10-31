#pragma once

#include <unordered_map>

#include "core/ir/ids.h"
#include "core/memory/tensor.h"

namespace ie {

class Workspace {
public:
    std::unordered_map<ValueId, Tensor>& tensors() { return tensors_; }
    const std::unordered_map<ValueId, Tensor>& tensors() const {
        return tensors_;
    }

private:
    std::unordered_map<ValueId, Tensor> tensors_;
};

}  // namespace ie
