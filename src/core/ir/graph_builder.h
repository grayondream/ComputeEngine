#pragma once

#include "core/ir/graph.h"

namespace ie {

class GraphBuilder {
public:
    // Predeclares the two default inputs "a" and "b" (both dynamic F32).
    GraphBuilder();

    GraphBuilder& add(int num_inputs);
    GraphBuilder& mul(double scalar);

    Graph build();

private:
    ValueId append_value(std::string name, bool is_input, bool is_output);
    NodeId append_node(OpType op, std::vector<ValueId> inputs, Attrs attrs,
                       bool mark_output);

    Graph graph_;
    std::vector<ValueId> input_ids_;
    std::optional<ValueId> previous_output_;
    int output_counter_ = 0;
};

}  // namespace ie
