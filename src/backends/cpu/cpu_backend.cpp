#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include "backends/cpu/cpu_allocator.h"
#include "backends/cpu/kernels/cpu_kernel_catalog.h"
#include "core/hardware/detector.h"
#include "core/runtime/executor.h"
#include "core/runtime/thread_pool.h"

namespace ie {
namespace {

constexpr std::int64_t kMaxWorkerThreads = 64;

std::size_t worker_count(const HardwareProfile& profile) {
    const std::int64_t requested =
        static_cast<std::int64_t>(profile.cu.num_cores) *
        static_cast<std::int64_t>(profile.cu.threads_per_core);
    const std::int64_t clamped =
        std::clamp<std::int64_t>(requested, 1, kMaxWorkerThreads);
    return static_cast<std::size_t>(clamped);
}

}  // namespace

CpuBackend::CpuBackend() : allocator_(std::make_shared<CpuAllocator>()) {}

CpuBackend::~CpuBackend() { (void)shutdown(); }

Status CpuBackend::init() {
    if (initialized_) {
        return {};
    }

    Result<HardwareProfile> detected = HardwareDetector::detect_cpu();
    if (!detected.has_value()) {
        return detected.error();
    }

    Result<CpuKernelCatalog::Selection> selected =
        CpuKernelCatalog::select(*detected);
    if (!selected.has_value()) {
        return selected.error();
    }

    std::unique_ptr<ThreadPool> pool;
    try {
        pool = std::make_unique<ThreadPool>(worker_count(*detected));
    } catch (const std::exception& error) {
        return Error::runtime(std::string("cpu thread pool creation failed: ") +
                              error.what());
    }

    profile_ = std::move(*detected);
    kernels_ = selected->fns;
    pool_ = std::move(pool);
    initialized_ = true;
    return {};
}

Status CpuBackend::shutdown() {
    // Stop workers before dropping the graph they may reference.
    pool_.reset();
    graph_.reset();
    kernels_ = {};
    initialized_ = false;
    return {};
}

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
    if (!initialized_) {
        return Error::invalid_argument("cpu backend is not initialized");
    }
    if (!graph_.has_value()) {
        return Error::invalid_argument("cpu backend has no compiled graph");
    }
    Executor executor;
    std::shared_ptr<Allocator> allocator =
        context.allocator ? context.allocator
                          : std::static_pointer_cast<Allocator>(allocator_);
    auto result = executor.run(*graph_, context.inputs, std::move(allocator),
                               &kernels_, pool_.get());
    if (!result.has_value()) {
        return result.error();
    }
    context.outputs = std::move(*result);
    return {};
}

Status CpuBackend::wait(Timeout /*timeout*/) { return {}; }

bool CpuBackend::is_idle() const { return true; }

BackendCaps CpuBackend::caps() const {
    BackendCaps caps;
    caps.supports_deterministic_kernels = true;
    caps.supports_batch_invariant = true;
    return caps;
}

PluginMeta CpuBackend::meta() const {
    return PluginMeta{"cpu", "0.1.0", "0.1", {"backend", "cpu"},
                      "synchronous P0 CPU backend"};
}

Result<HardwareProfile> CpuBackend::hardware_profile() const {
    return profile_;
}

Result<MemoryPolicy> CpuBackend::memory_policy() const {
    return MemoryPolicy{};
}

}  // namespace ie
