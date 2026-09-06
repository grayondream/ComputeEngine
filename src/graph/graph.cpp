#include "compute_engine/graph/graph.h"
#include <algorithm>
#include <unordered_set>
#include <sstream>
#include <queue>
#include "compute_engine/platform/logger.h"

namespace compute_engine {
namespace graph {

NodeDef* GraphDef::AddNode(const std::string& name, OpType op_type) {
    auto node = std::make_unique<NodeDef>();
    node->name = name;
    node->op_type = op_type;
    node_index_[name] = nodes_.size();
    nodes_.push_back(std::move(node));
    return nodes_.back().get();
}

NodeDef* GraphDef::GetNode(const std::string& name) {
    auto it = node_index_.find(name);
    if (it != node_index_.end()) return nodes_[it->second].get();
    return nullptr;
}

const NodeDef* GraphDef::GetNode(const std::string& name) const {
    auto it = node_index_.find(name);
    if (it != node_index_.end()) return nodes_[it->second].get();
    return nullptr;
}

void GraphDef::AddInput(const std::string& name, const Shape& shape, DType dtype) {
    inputs_.push_back(name);
    auto tensor = std::make_shared<TensorDef>();
    tensor->name = name;
    tensor->shape = shape;
    tensor->dtype = dtype;
    tensors_[name] = tensor;
}

void GraphDef::AddOutput(const std::string& name) {
    outputs_.push_back(name);
}

void GraphDef::SetConstant(const std::string& name, TensorPtr tensor) {
    auto def = std::make_shared<TensorDef>();
    def->name = name;
    def->shape = tensor->shape();
    def->dtype = tensor->dtype();
    def->data = tensor->data();
    def->is_constant = true;
    tensors_[name] = def;
}

TensorDefPtr GraphDef::GetTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return it->second;
    return nullptr;
}

void GraphDef::RegisterTensor(TensorDefPtr tensor) {
    tensors_[tensor->name] = tensor;
}

Status GraphDef::TopologicalSort(std::vector<NodeDef*>& sorted) const {
    // Build map: tensor_name -> producer_node_name
    std::unordered_map<std::string, std::string> tensor_producer;
    for (auto& node : nodes_) {
        for (auto& output : node->outputs) {
            tensor_producer[output] = node->name;
        }
    }

    std::unordered_map<std::string, int> in_degree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;

    for (auto& node : nodes_) {
        in_degree[node->name] = 0;
    }

    for (auto& node : nodes_) {
        for (auto& input : node->inputs) {
            auto producer_it = tensor_producer.find(input);
            if (producer_it != tensor_producer.end()) {
                dependents[producer_it->second].push_back(node->name);
                in_degree[node->name]++;
            }
        }
    }

    std::queue<std::string> queue;
    for (auto& [name, deg] : in_degree) {
        if (deg == 0) queue.push(name);
    }

    sorted.clear();
    while (!queue.empty()) {
        auto name = queue.front();
        queue.pop();
        auto* node = const_cast<GraphDef*>(this)->GetNode(name);
        if (node) sorted.push_back(node);

        for (auto& dep : dependents[name]) {
            in_degree[dep]--;
            if (in_degree[dep] == 0) queue.push(dep);
        }
    }

    if (sorted.size() != nodes_.size()) {
        return Status::InvalidArgument("Graph has cycles or disconnected nodes");
    }
    return Status::OK();
}

Status GraphDef::Validate() const {
    for (auto& node : nodes_) {
        if (node->inputs.empty() && node->op_type != OpType::CUSTOM) {
            // Allow nodes with no inputs (constants, etc.)
        }

        for (auto& input : node->inputs) {
            if (tensors_.find(input) == tensors_.end() &&
                node_index_.find(input) == node_index_.end()) {
                // Input might be a graph input
                bool found = false;
                for (auto& gi : inputs_) {
                    if (gi == input) { found = true; break; }
                }
                if (!found) {
                    LOGW("Node '{}' references unknown tensor '{}'", node->name, input);
                }
            }
        }
    }
    return Status::OK();
}

Status GraphDef::InferShapes() {
    // Basic shape inference - propagate shapes through the graph
    // This is a simplified version; full inference would use OpSchema
    for (auto& node : nodes_) {
        if (node->op_type == OpType::RESHAPE) {
            auto* shape_attr = node->GetAttr("shape");
            if (shape_attr && shape_attr->type == AttributeType::SHAPE) {
                if (!node->inputs.empty()) {
                    auto it = tensors_.find(node->inputs[0]);
                    if (it != tensors_.end()) {
                        auto out_tensor = std::make_shared<TensorDef>();
                        out_tensor->name = node->outputs.empty() ? node->name + "_out" : node->outputs[0];
                        out_tensor->shape = shape_attr->shape;
                        out_tensor->dtype = it->second->dtype;
                        tensors_[out_tensor->name] = out_tensor;
                    }
                }
            }
        } else if (node->op_type == OpType::TRANSPOSE) {
            auto* perm_attr = node->GetAttr("perm");
            if (perm_attr && perm_attr->type == AttributeType::INTS && !node->inputs.empty()) {
                auto it = tensors_.find(node->inputs[0]);
                if (it != tensors_.end()) {
                    auto out_tensor = std::make_shared<TensorDef>();
                    out_tensor->name = node->outputs.empty() ? node->name + "_out" : node->outputs[0];
                    auto& perm = perm_attr->ints;
                    std::vector<int64_t> new_dims(perm.size());
                    for (size_t i = 0; i < perm.size(); ++i)
                        new_dims[i] = it->second->shape[perm[i]];
                    out_tensor->shape = Shape(new_dims);
                    out_tensor->dtype = it->second->dtype;
                    tensors_[out_tensor->name] = out_tensor;
                }
            }
        }
    }
    return Status::OK();
}

std::string GraphDef::ToDot() const {
    std::ostringstream ss;
    ss << "digraph " << name_ << " {\n";
    ss << "  rankdir=TB;\n";
    ss << "  node [shape=record];\n\n";

    for (auto& node : nodes_) {
        std::string label = node->name + "\\n" + OpTypeToString(node->op_type);
        ss << "  \"" << node->name << "\" [label=\"" << label << "\"];\n";
    }

    ss << "\n";
    for (auto& node : nodes_) {
        for (auto& input : node->inputs) {
            // Find which node produces this tensor
            for (auto& producer : nodes_) {
                for (auto& out : producer->outputs) {
                    if (out == input) {
                        ss << "  \"" << producer->name << "\" -> \"" << node->name
                           << "\" [label=\"" << input << "\"];\n";
                    }
                }
            }
        }
    }

    ss << "}\n";
    return ss.str();
}

std::string GraphDef::ToString() const {
    std::ostringstream ss;
    ss << "Graph(" << name_ << "): " << nodes_.size() << " nodes, "
       << inputs_.size() << " inputs, " << outputs_.size() << " outputs\n";
    for (auto& node : nodes_) {
        ss << "  " << node->name << " [" << OpTypeToString(node->op_type) << "]"
           << " in=" << node->inputs.size() << " out=" << node->outputs.size() << "\n";
    }
    return ss.str();
}

} // namespace graph
} // namespace compute_engine
