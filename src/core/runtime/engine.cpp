#include "core/runtime/engine.h"

#include <chrono>
#include <utility>

#include "backends/cpu/cpu_backend.h"
#include "core/backend/backend_registry.h"
#include "core/runtime/execution_context.h"
#include "core/scheduler/plan.h"

namespace ie {

Engine::Engine(EngineConfig config)
    : config_(std::move(config)),
      allocator_(std::make_shared<MallocAllocator>()) {
    Status valid = validate(config_);
    if (!valid.has_value()) {
        error_ = valid.error();
        return;
    }

    auto& registry = BackendRegistry::instance();
    if (!registry.contains(BackendKind::CPU)) {
        (void)registry.register_backend(
            BackendKind::CPU, [] { return std::make_unique<CpuBackend>(); });
    }
    auto backend = registry.create(BackendKind::CPU);
    if (!backend.has_value()) {
        error_ = backend.error();
        return;
    }
    if (!backend.value()) {
        error_ = Error::runtime("cpu backend factory returned null");
        return;
    }
    backend_ = std::move(*backend);
    Status initialized = backend_->init();
    if (!initialized.has_value()) {
        error_ = initialized.error();
        backend_.reset();
    }
}

Engine::~Engine() {
    if (backend_) {
        (void)backend_->shutdown();
    }
}

Status Engine::load(const Graph& graph) {
    if (error_.has_value()) {
        return *error_;
    }
    if (!backend_) {
        return Error::invalid_argument("no backend available");
    }
    graph_ = graph;
    Status compiled = backend_->compile(*graph_);
    if (!compiled.has_value()) {
        return compiled.error();
    }
    loaded_ = true;
    return {};
}

Result<std::vector<Tensor>> Engine::infer(
    const std::unordered_map<std::string, Tensor>& inputs) {
    if (error_.has_value()) {
        return *error_;
    }
    if (!loaded_ || !graph_.has_value()) {
        return Error::invalid_argument("engine has no loaded graph");
    }

    ExecutionContext context;
    context.allocator = allocator_;
    context.inputs = inputs;

    BatchPlan plan;
    PrefillItem item;
    item.request = 0;
    plan.prefills.push_back(std::move(item));
    for (const auto& [name, tensor] : inputs) {
        (void)name;
        plan.token_budget += static_cast<std::size_t>(tensor.numel());
    }

    Status executed = backend_->execute_async(plan, context);
    if (!executed.has_value()) {
        return executed.error();
    }
    Status waited = backend_->wait(std::chrono::milliseconds(1000));
    if (!waited.has_value()) {
        return waited.error();
    }
    return std::move(context.outputs);
}

}  // namespace ie
