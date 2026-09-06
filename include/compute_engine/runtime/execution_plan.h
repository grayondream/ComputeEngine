#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/tensor.h"
#include "compute_engine/op/op.h"
#include "compute_engine/graph/graph.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace compute_engine {
namespace runtime {

struct MemorySchedule {
    std::string tensor_name;
    size_t offset = 0;
    size_t size = 0;
    bool is_temporary = false;
};

struct KernelNode {
    std::string name;
    OpType op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<graph::Attribute> attrs;
    int stage = 0;
};

class ExecutionPlan {
public:
    ExecutionPlan() = default;
    explicit ExecutionPlan(const std::string& graph_name) : graph_name_(graph_name) {}

    const std::string& graph_name() const { return graph_name_; }

    void AddKernelNode(const KernelNode& node);
    size_t num_nodes() const { return nodes_.size(); }
    const std::vector<KernelNode>& nodes() const { return nodes_; }

    void SetMemorySchedule(const std::vector<MemorySchedule>& schedule);
    const std::vector<MemorySchedule>& memory_schedule() const { return memory_schedule_; }

    Status Validate() const;
    std::string ToString() const;

private:
    std::string graph_name_;
    std::vector<KernelNode> nodes_;
    std::vector<MemorySchedule> memory_schedule_;
};

class GraphCompiler {
public:
    static GraphCompiler& Instance();

    Status Compile(const graph::GraphDef& graph, ExecutionPlan* plan);

    ExecutionPlan* GetCachedPlan(const std::string& key);
    void CachePlan(const std::string& key, std::unique_ptr<ExecutionPlan> plan);

private:
    std::unordered_map<std::string, std::unique_ptr<ExecutionPlan>> cache_;
};

class GraphExecutor {
public:
    Status Compile(const graph::GraphDef& graph);
    Status Execute(const std::unordered_map<std::string, TensorPtr>& inputs,
                   std::unordered_map<std::string, TensorPtr>& outputs);

    const ExecutionPlan* plan() const { return plan_.get(); }

private:
    std::unique_ptr<ExecutionPlan> plan_;
    std::unordered_map<std::string, TensorPtr> constants_;
};

} // namespace runtime
} // namespace compute_engine
