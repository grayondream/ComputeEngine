#include "compute_engine/runtime/execution_plan.h"
#include "compute_engine/platform/logger.h"
#include <sstream>
#include <algorithm>

namespace compute_engine {
namespace runtime {

// ─── ExecutionPlan ───
void ExecutionPlan::AddKernelNode(const KernelNode& node) {
    nodes_.push_back(node);
}

void ExecutionPlan::SetMemorySchedule(const std::vector<MemorySchedule>& schedule) {
    memory_schedule_ = schedule;
}

Status ExecutionPlan::Validate() const {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& node = nodes_[i];
        if (node.name.empty()) {
            return Status::InvalidArgument("KernelNode at index " + std::to_string(i) + " has empty name");
        }
        for (auto& input : node.inputs) {
            bool found = false;
            for (size_t j = 0; j < i; ++j) {
                for (auto& out : nodes_[j].outputs) {
                    if (out == input) { found = true; break; }
                }
                if (found) break;
            }
            if (!found) {
                // Could be a graph input or constant - OK
            }
        }
    }
    return Status::OK();
}

std::string ExecutionPlan::ToString() const {
    std::ostringstream ss;
    ss << "ExecutionPlan(" << graph_name_ << "): " << nodes_.size() << " kernel nodes\n";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& node = nodes_[i];
        ss << "  [" << i << "] " << node.name
           << " op=" << OpTypeToString(node.op_type)
           << " stage=" << node.stage
           << " in=" << node.inputs.size()
           << " out=" << node.outputs.size() << "\n";
    }
    if (!memory_schedule_.empty()) {
        ss << "  Memory schedule: " << memory_schedule_.size() << " regions\n";
    }
    return ss.str();
}

// ─── GraphCompiler ───
GraphCompiler& GraphCompiler::Instance() {
    static GraphCompiler instance;
    return instance;
}

Status GraphCompiler::Compile(const graph::GraphDef& graph, ExecutionPlan* plan) {
    // Check cache first
    std::string cache_key = graph.name();
    auto* cached = GetCachedPlan(cache_key);
    if (cached) {
        *plan = *cached;
        LOGI("GraphCompiler: cache hit for '{}'", cache_key);
        return Status::OK();
    }

    // Topological sort
    std::vector<graph::NodeDef*> sorted;
    auto st = graph.TopologicalSort(sorted);
    if (!st.ok()) return st;

    // Build execution plan
    auto exec_plan = std::make_unique<ExecutionPlan>(graph.name());

    for (auto* node : sorted) {
        KernelNode knode;
        knode.name = node->name;
        knode.op_type = node->op_type;
        knode.inputs = node->inputs;
        knode.outputs = node->outputs;
        knode.attrs = node->attrs;
        knode.stage = 0;
        exec_plan->AddKernelNode(knode);
    }

    // Assign stages based on dependencies (simple parallelism detection)
    std::unordered_map<std::string, int> tensor_stage;
    for (auto& gi : graph.inputs()) tensor_stage[gi] = 0;

    for (size_t idx = 0; idx < exec_plan->num_nodes(); ++idx) {
        auto& knode = const_cast<KernelNode&>(exec_plan->nodes()[idx]);
        int max_input_stage = 0;
        for (auto& input : knode.inputs) {
            auto it = tensor_stage.find(input);
            if (it != tensor_stage.end())
                max_input_stage = std::max(max_input_stage, it->second);
        }
        knode.stage = max_input_stage;
        for (auto& output : knode.outputs) {
            tensor_stage[output] = max_input_stage + 1;
        }
    }

    // Simple memory schedule: allocate all tensors linearly
    std::vector<MemorySchedule> schedule;
    size_t offset = 0;
    for (auto& knode : exec_plan->nodes()) {
        for (auto& output : knode.outputs) {
            MemorySchedule ms;
            ms.tensor_name = output;
            ms.offset = offset;
            ms.size = 0; // Would be computed from tensor shapes in full impl
            ms.is_temporary = true;
            schedule.push_back(ms);
            offset += 1024; // Placeholder
        }
    }
    exec_plan->SetMemorySchedule(schedule);

    st = exec_plan->Validate();
    if (!st.ok()) return st;

    LOGI("GraphCompiler: compiled '{}' into {} kernel nodes, {} stages",
         graph.name(), exec_plan->num_nodes(),
         exec_plan->nodes().empty() ? 0 :
         exec_plan->nodes().back().stage + 1);

    CachePlan(cache_key, std::move(exec_plan));
    *plan = *GetCachedPlan(cache_key);
    return Status::OK();
}

ExecutionPlan* GraphCompiler::GetCachedPlan(const std::string& key) {
    auto it = cache_.find(key);
    return (it != cache_.end()) ? it->second.get() : nullptr;
}

void GraphCompiler::CachePlan(const std::string& key, std::unique_ptr<ExecutionPlan> plan) {
    cache_[key] = std::move(plan);
}

// ─── GraphExecutor ───
Status GraphExecutor::Compile(const graph::GraphDef& graph) {
    plan_ = std::make_unique<ExecutionPlan>();
    return GraphCompiler::Instance().Compile(graph, plan_.get());
}

Status GraphExecutor::Execute(
    const std::unordered_map<std::string, TensorPtr>& inputs,
    std::unordered_map<std::string, TensorPtr>& outputs) {
    if (!plan_) return Status::InvalidState("Graph not compiled");

    LOGI("GraphExecutor: executing plan with {} nodes", plan_->num_nodes());

    // Placeholder: in full implementation, iterate through plan nodes
    // and dispatch each kernel with proper tensor arguments
    for (auto& node : plan_->nodes()) {
        LOGD("  Execute: {} [{}]", node.name, OpTypeToString(node.op_type));
    }

    return Status::OK();
}

} // namespace runtime
} // namespace compute_engine
