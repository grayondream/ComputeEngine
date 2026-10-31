#pragma once

#include <chrono>
#include <cstddef>
#include <memory>

#include "core/backend/backend_caps.h"
#include "core/base/error.h"
#include "core/base/types.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/memory/allocator.h"
#include "core/memory/policy.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct BatchPlan;
struct ExecutionContext;

using Timeout = std::chrono::milliseconds;

// Backend interface (docs/readme.md §3.3). P0 implements the synchronous CPU
// backend; GPU backends stay interface-only.
class Backend {
public:
    virtual ~Backend() = default;

    virtual Status init() = 0;
    virtual Status shutdown() = 0;
    virtual Device device() const = 0;

    virtual Result<BufferHandle> alloc(std::size_t bytes) = 0;
    virtual void free(BufferHandle handle) = 0;
    virtual Status copy_to_device(BufferHandle dst, const void* src,
                                  std::size_t bytes) = 0;
    virtual Status copy_to_host(void* dst, BufferHandle src,
                                std::size_t bytes) = 0;
    virtual Status copy_d2d(BufferHandle dst, BufferHandle src,
                            std::size_t bytes) = 0;

    virtual Status compile(Graph& graph) = 0;
    virtual Status execute_async(const BatchPlan& plan,
                                 ExecutionContext& context) = 0;
    virtual Status wait(Timeout timeout) = 0;
    virtual bool is_idle() const = 0;

    virtual BackendCaps caps() const = 0;
    virtual PluginMeta meta() const = 0;

    virtual Result<HardwareProfile> hardware_profile() const = 0;
    virtual Result<MemoryPolicy> memory_policy() const = 0;
};

}  // namespace ie
