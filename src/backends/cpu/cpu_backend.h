#pragma once

#include <memory>

#include "core/backend/backend.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/ops/kernel_fns.h"

namespace ie {

class CpuAllocator;
class ThreadPool;

// CPU backend: capability-driven kernels executed on a deterministic pool.
class CpuBackend final : public Backend {
public:
    CpuBackend();
    ~CpuBackend() override;

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
    HardwareProfile profile_;
    std::unique_ptr<ThreadPool> pool_;
    std::shared_ptr<CpuAllocator> allocator_;
    KernelFns kernels_;
    bool initialized_{false};
    std::unique_ptr<Graph> graph_;
};

}  // namespace ie
