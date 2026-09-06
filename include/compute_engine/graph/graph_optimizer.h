#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/graph/graph.h"
#include <vector>
#include <string>

namespace compute_engine {
namespace graph {

class GraphOptimizerPass {
public:
    virtual ~GraphOptimizerPass() = default;
    virtual std::string name() const = 0;
    virtual Status Run(GraphDef& graph) = 0;
};

class ConstFoldPass : public GraphOptimizerPass {
public:
    std::string name() const override { return "ConstFold"; }
    Status Run(GraphDef& graph) override;
};

class DeadCodeElimPass : public GraphOptimizerPass {
public:
    std::string name() const override { return "DeadCodeElim"; }
    Status Run(GraphDef& graph) override;
};

class OpFusionPass : public GraphOptimizerPass {
public:
    std::string name() const override { return "OpFusion"; }
    Status Run(GraphDef& graph) override;
};

class GraphOptimizer {
public:
    void AddPass(std::unique_ptr<GraphOptimizerPass> pass);
    Status Optimize(GraphDef& graph, int max_iterations = 3);

    static GraphOptimizer Default();

private:
    std::vector<std::unique_ptr<GraphOptimizerPass>> passes_;
};

} // namespace graph
} // namespace compute_engine
