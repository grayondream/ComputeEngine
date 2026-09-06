#include "compute_engine/graph/graph_optimizer.h"
#include "compute_engine/platform/logger.h"
#include <unordered_set>
#include <algorithm>

namespace compute_engine {
namespace graph {

// ─── Constant Folding ───
// Folds nodes whose inputs are all constants into computed constant tensors.
// This is a simplified version that handles elementwise ops on constants.
Status ConstFoldPass::Run(GraphDef& graph) {
    bool changed = true;
    int iterations = 0;

    while (changed && iterations < 10) {
        changed = false;
        iterations++;

        std::vector<std::string> to_remove;

        for (auto& node : graph.nodes()) {
            if (node->op_type == OpType::RESHAPE ||
                node->op_type == OpType::TRANSPOSE) {
                // Check if all inputs are constants
                bool all_const = true;
                for (auto& input : node->inputs) {
                    auto tensor = graph.GetTensor(input);
                    if (!tensor || !tensor->is_constant) {
                        all_const = false;
                        break;
                    }
                }
                if (!all_const) continue;

                // For reshape/transpose on constants, compute result at graph level
                // (simplified: just mark the output as constant if input is constant)
                if (!node->outputs.empty()) {
                    auto input_tensor = graph.GetTensor(node->inputs[0]);
                    if (input_tensor && input_tensor->is_constant) {
                        auto out_tensor = std::make_shared<TensorDef>();
                        out_tensor->name = node->outputs[0];
                        out_tensor->is_constant = true;
                        out_tensor->dtype = input_tensor->dtype;

                        if (node->op_type == OpType::RESHAPE) {
                            auto* shape_attr = node->GetAttr("shape");
                            if (shape_attr && shape_attr->type == AttributeType::SHAPE) {
                                out_tensor->shape = shape_attr->shape;
                            }
                        } else {
                            out_tensor->shape = input_tensor->shape;
                        }

                        out_tensor->data = input_tensor->data;
                        graph.RegisterTensor(out_tensor);
                        to_remove.push_back(node->name);
                        changed = true;
                        LOGI("ConstFold: folded {} -> {}", node->inputs[0], node->outputs[0]);
                    }
                }
            }
        }

        // Remove folded nodes (reverse order to maintain indices)
        std::sort(to_remove.begin(), to_remove.end());
        for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
            auto& nodes = const_cast<std::vector<std::unique_ptr<NodeDef>>&>(graph.nodes());
            auto nit = std::find_if(nodes.begin(), nodes.end(),
                [&](auto& n) { return n->name == *it; });
            if (nit != nodes.end()) nodes.erase(nit);
        }
    }

    return Status::OK();
}

// ─── Dead Code Elimination ───
// Removes nodes whose outputs are not used by any other node or graph output.
Status DeadCodeElimPass::Run(GraphDef& graph) {
    // Collect all tensors that are used
    std::unordered_set<std::string> used_tensors;

    // Mark graph outputs as used
    for (auto& output : graph.outputs()) {
        used_tensors.insert(output);
    }

    // Mark all node outputs as used if they feed into other nodes
    for (auto& node : graph.nodes()) {
        for (auto& input : node->inputs) {
            used_tensors.insert(input);
        }
    }

    // Find dead nodes (whose outputs are not in used_tensors)
    std::vector<std::string> to_remove;
    for (auto& node : graph.nodes()) {
        if (node->op_type == OpType::CUSTOM) continue;

        bool dead = true;
        for (auto& output : node->outputs) {
            if (used_tensors.count(output) > 0) {
                dead = false;
                break;
            }
        }

        // Also check if node has side effects (attention, etc.)
        if (node->op_type == OpType::ATTENTION ||
            node->op_type == OpType::KV_CACHE_SAVE) {
            dead = false;
        }

        if (dead) {
            to_remove.push_back(node->name);
        }
    }

    // Remove dead nodes
    for (auto& name : to_remove) {
        auto& nodes = const_cast<std::vector<std::unique_ptr<NodeDef>>&>(graph.nodes());
        auto it = std::find_if(nodes.begin(), nodes.end(),
            [&](auto& n) { return n->name == name; });
        if (it != nodes.end()) {
            LOGI("DeadCodeElim: removed {}", name);
            nodes.erase(it);
        }
    }

    return Status::OK();
}

// ─── Op Fusion ───
// Fuses patterns like Add+ReLU -> AddReLU, MatMul+Add -> Linear
Status OpFusionPass::Run(GraphDef& graph) {
    // Simplified fusion: find Add -> ReLU pattern and mark for fusion
    // Full implementation would rewrite the graph
    std::unordered_set<std::string> consumed;

    for (auto& node : graph.nodes()) {
        for (auto& input : node->inputs) {
            consumed.insert(input);
        }
    }

    // Placeholder for future fusion patterns
    // A real implementation would detect fusible patterns and rewrite the graph
    return Status::OK();
}

// ─── Graph Optimizer ───
void GraphOptimizer::AddPass(std::unique_ptr<GraphOptimizerPass> pass) {
    passes_.push_back(std::move(pass));
}

Status GraphOptimizer::Optimize(GraphDef& graph, int max_iterations) {
    for (int i = 0; i < max_iterations; ++i) {
        bool any_changed = false;
        for (auto& pass : passes_) {
            size_t before = graph.num_nodes();
            auto st = pass->Run(graph);
            if (!st.ok()) {
                LOGW("Optimization pass '{}' failed: {}", pass->name(), st.ToString());
                return st;
            }
            if (graph.num_nodes() != before) any_changed = true;
            LOGI("Pass '{}' completed: {} -> {} nodes",
                 pass->name(), before, graph.num_nodes());
        }
        if (!any_changed) break;
    }
    return Status::OK();
}

GraphOptimizer GraphOptimizer::Default() {
    GraphOptimizer opt;
    opt.AddPass(std::make_unique<ConstFoldPass>());
    opt.AddPass(std::make_unique<DeadCodeElimPass>());
    opt.AddPass(std::make_unique<OpFusionPass>());
    return opt;
}

} // namespace graph
} // namespace compute_engine
