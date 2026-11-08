#include "core/runtime/executor.h"

#include <unordered_map>
#include <utility>

#include "core/runtime/execution_context.h"

namespace ie {

Result<std::vector<Tensor>> Executor::run(
    const Graph& graph,
    const std::unordered_map<std::string, Tensor>& inputs,
    std::shared_ptr<Allocator> allocator, const KernelFns* kernels,
    ThreadPool* pool) {
    if (!allocator) {
        return Error::invalid_argument("executor requires an allocator");
    }
    std::unordered_map<ValueId, Tensor> values;

    for (const Value& value : graph.values) {
        if (!value.is_input) {
            continue;
        }
        auto it = inputs.find(value.name);
        if (it == inputs.end()) {
            return Error::invalid_argument("missing input: " + value.name);
        }
        if (it->second.dtype != DType::F32) {
            return Error::invalid_argument("P0 supports F32 inputs only");
        }
        if (it->second.buffer == nullptr) {
            return Error::invalid_argument("input has no buffer: " + value.name);
        }
        values.emplace(value.id, it->second);
    }

    ExecutionContext context;
    context.allocator = std::move(allocator);
    context.kernels = kernels;
    context.thread_pool = pool;

    for (const Node& node : graph.nodes) {
        auto op = registry_.create(node.op);
        if (!op.has_value()) {
            return op.error();
        }
        if (!op.value()) {
            return Error::runtime("op factory returned null");
        }
        Status configured = op.value()->configure(node.attrs);
        if (!configured.has_value()) {
            return configured.error();
        }

        std::vector<Tensor*> in_ptrs;
        in_ptrs.reserve(node.inputs.size());
        for (ValueId id : node.inputs) {
            auto it = values.find(id);
            if (it == values.end()) {
                return Error::invalid_argument("missing node input value");
            }
            in_ptrs.push_back(&it->second);
        }

        std::vector<Tensor> out_storage(node.outputs.size());
        std::vector<Tensor*> out_ptrs;
        out_ptrs.reserve(out_storage.size());
        for (Tensor& tensor : out_storage) {
            out_ptrs.push_back(&tensor);
        }

        Status forwarded = op.value()->forward(in_ptrs, out_ptrs, context);
        if (!forwarded.has_value()) {
            return forwarded.error();
        }

        for (std::size_t i = 0; i < node.outputs.size(); ++i) {
            values.emplace(node.outputs[i], std::move(out_storage[i]));
        }
    }

    std::vector<Tensor> outputs;
    for (const Value& value : graph.values) {
        if (!value.is_output) {
            continue;
        }
        auto it = values.find(value.id);
        if (it == values.end()) {
            return Error::invalid_argument("output value was not produced");
        }
        outputs.push_back(it->second);
    }
    if (outputs.empty()) {
        return Error::invalid_argument("graph has no outputs");
    }
    return outputs;
}

}  // namespace ie
