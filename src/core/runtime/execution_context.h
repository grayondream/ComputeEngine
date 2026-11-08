#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/base/types.h"
#include "core/memory/allocator.h"
#include "core/memory/tensor.h"
#include "core/ops/kernel_fns.h"

namespace ie {

class ThreadPool;

struct ExecutionContext {
    Device device{Device::CPU};
    std::shared_ptr<Allocator> allocator;
    std::unordered_map<std::string, Tensor> inputs;
    std::vector<Tensor> outputs;
    void* stream{nullptr};
    // Optional capability-selected kernels and the pool used to spread
    // independent work. Null means "use the scalar reference implementation".
    const KernelFns* kernels{nullptr};
    ThreadPool* thread_pool{nullptr};
};

}  // namespace ie
