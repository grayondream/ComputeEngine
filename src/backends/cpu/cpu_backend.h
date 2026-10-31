#pragma once

#include <optional>

#include "core/backend/backend.h"
#include "core/ir/graph.h"
#include "core/memory/malloc_allocator.h"

namespace ie {

// Synchronous P0 CPU backend.
class CpuBackend final : public Backend {
public:
    Status init() override;
    Status shutdown() override;
    Device device() const override;

    Result<BufferHandle> alloc(std::size_t bytes) override;
    void free(BufferHandle handle) override;
    Status copy_to_device(BufferHandle dst, const void* src,
                          std::size_t bytes) override;
    Status copy_to_host(void* dst, BufferHandle src,
                        std::size_t bytes) override;
    Status copy_d2d(BufferHandle dst, BufferHandle src,
                    std::size_t bytes) override;

    Status compile(Graph& graph) override;
    Status execute_async(const BatchPlan& plan,
                         ExecutionContext& context) override;
    Status wait(Timeout timeout) override;
    bool is_idle() const override;

    BackendCaps caps() const override;
    PluginMeta meta() const override;

    Result<HardwareProfile> hardware_profile() const override;
    Result<MemoryPolicy> memory_policy() const override;

private:
    MallocAllocator allocator_;
    std::optional<Graph> graph_;
};

}  // namespace ie
