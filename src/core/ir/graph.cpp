#include "core/ir/graph_builder.h"

#include <algorithm>
#include <utility>

namespace ie {

GraphBuilder::GraphBuilder() {
    input_ids_.push_back(append_value("a", true, false));
    input_ids_.push_back(append_value("b", true, false));
}

ValueId GraphBuilder::append_value(std::string name, bool is_input,
                                   bool is_output) {
    Value value;
    value.id = static_cast<ValueId>(graph_.values.size());
    value.name = std::move(name);
    value.shape = Shape{{-1}};
    value.dtype = DType::F32;
    value.is_input = is_input;
    value.is_output = is_output;
    graph_.values.push_back(std::move(value));
    return graph_.values.back().id;
}

NodeId GraphBuilder::append_node(OpType op, std::vector<ValueId> inputs,
                                 Attrs attrs, bool mark_output) {
    const NodeId node_id = static_cast<NodeId>(graph_.nodes.size());
    const ValueId output_id = append_value(
        "t" + std::to_string(output_counter_++), false, mark_output);

    Node node;
    node.id = node_id;
    node.op = op;
    node.inputs = std::move(inputs);
    node.outputs = {output_id};
    node.attrs = std::move(attrs);
    graph_.nodes.push_back(std::move(node));

    for (ValueId input : graph_.nodes.back().inputs) {
        graph_.edges.push_back(Edge{input, std::nullopt, node_id});
    }
    previous_output_ = output_id;
    return node_id;
}

GraphBuilder& GraphBuilder::add(int num_inputs) {
    const int count = std::clamp(num_inputs, 0,
                                 static_cast<int>(input_ids_.size()));
    std::vector<ValueId> inputs(input_ids_.begin(), input_ids_.begin() + count);
    append_node(OpType::Add, std::move(inputs), {}, false);
    return *this;
}

GraphBuilder& GraphBuilder::mul(double scalar) {
    Attrs attrs;
    attrs.emplace("scalar", scalar);
    std::vector<ValueId> inputs;
    if (previous_output_.has_value()) {
        inputs.push_back(*previous_output_);
    }
    append_node(OpType::Mul, std::move(inputs), std::move(attrs), true);
    return *this;
}

Graph GraphBuilder::build() { return std::move(graph_); }

Result<BoundGraph> Graph::bind(const ShapeMap& shapes) const {
    BoundGraph bound;
    for (const Value& value : values) {
        if (!value.is_input) {
            continue;
        }
        auto it = shapes.find(value.name);
        if (it == shapes.end()) {
            return Error::invalid_argument("missing shape for input: " +
                                           value.name);
        }
        bound.shapes.emplace(value.id, it->second);
    }
    return bound;
}

Result<Graph> Graph::build_backward(
    const std::vector<ValueId>& /*loss_inputs*/) const {
    return Error::unsupported("backward not implemented in P0");
}

const Value* Graph::find_value(ValueId id) const {
    for (const Value& value : values) {
        if (value.id == id) {
            return &value;
        }
    }
    return nullptr;
}

const Value* Graph::find_value(std::string_view name) const {
    for (const Value& value : values) {
        if (value.name == name) {
            return &value;
        }
    }
    return nullptr;
}

const Value* Graph::input(std::string_view name) const {
    const Value* value = find_value(name);
    if (value != nullptr && value->is_input) {
        return value;
    }
    return nullptr;
}

}  // namespace ie
