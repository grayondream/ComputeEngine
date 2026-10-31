#include "backends/cpu/cpu_backend.h"

#include <cstring>
#include <utility>

#include "core/hardware/detector.h"
#include "core/runtime/executor.h"

namespace ie {

Status CpuBackend::init() { return {}; }

Status CpuBackend::shutdown() { return {}; }

Device CpuBackend::device() const { return Device::CPU; }

Result<BufferHandle> CpuBackend::alloc(std::size_t bytes) {
    return allocator_->alloc(bytes, MemoryKind::HOST);
}

void CpuBackend::free(BufferHandle handle) { allocator_->free(handle); }

Status CpuBackend::copy_to_device(BufferHandle dst, const void* src,
                                  std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    if (dst == nullptr || src == nullptr) {
        return Error::invalid_argument("null copy buffer");
    }
    std::memcpy(dst, src, bytes);
    return {};
}

Status CpuBackend::copy_to_host(void* dst, BufferHandle src,
                                std::size_t bytes) {
    return copy_to_device(dst, src, bytes);
}

Status CpuBackend::copy_d2d(BufferHandle dst, BufferHandle src,
                            std::size_t bytes) {
    return copy_to_device(dst, src, bytes);
}

Status CpuBackend::compile(Graph& graph) {
    graph_ = graph;
    return {};
}

Status CpuBackend::execute_async(const BatchPlan& /*plan*/,
                                 ExecutionContext& context) {
    if (!graph_.has_value()) {
        return Error::invalid_argument("cpu backend has no compiled graph");
    }
    Executor executor;
    std::shared_ptr<Allocator> allocator =
        context.allocator ? context.allocator
                          : std::static_pointer_cast<Allocator>(allocator_);
    auto result =
        executor.run(*graph_, context.inputs, std::move(allocator));
    if (!result.has_value()) {
        return result.error();
    }
    context.outputs = std::move(*result);
    return {};
}

Status CpuBackend::wait(Timeout /*timeout*/) { return {}; }

bool CpuBackend::is_idle() const { return true; }

BackendCaps CpuBackend::caps() const { return {}; }

PluginMeta CpuBackend::meta() const {
    return PluginMeta{"cpu", "0.1.0", "0.1", {"backend", "cpu"},
                      "synchronous P0 CPU backend"};
}

Result<HardwareProfile> CpuBackend::hardware_profile() const {
    return HardwareDetector::detect_cpu();
}

Result<MemoryPolicy> CpuBackend::memory_policy() const {
    return MemoryPolicy{};
}

}  // namespace ie
