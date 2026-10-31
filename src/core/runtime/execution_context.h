#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/base/types.h"
#include "core/memory/allocator.h"
#include "core/memory/tensor.h"

namespace ie {

struct ExecutionContext {
    Device device{Device::CPU};
    Allocator* allocator{nullptr};
    std::unordered_map<std::string, Tensor> inputs;
    std::vector<Tensor> outputs;
    void* stream{nullptr};
};

}  // namespace ie
