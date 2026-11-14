#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/ir/attr.h"
#include "core/ir/ids.h"
#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/memory/tensor.h"

namespace ie {

using ShapeMap = std::unordered_map<std::string, Shape>;

struct BoundGraph {
    std::unordered_map<ValueId, Shape> shapes;
};

struct Value {
    ValueId id{kInvalidValue};
    std::string name;
    Shape shape;
    DType dtype{DType::F32};
    bool is_input{false};
    bool is_output{false};
};

struct Node {
    NodeId id{kInvalidNode};
    OpType op{OpType::Add};
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    Attrs attrs;
    std::optional<Device> placed_device;
    ModelKind kind{ModelKind::UNKNOWN};
};

struct Edge {
    ValueId value{kInvalidValue};
    std::optional<NodeId> from;
    std::optional<NodeId> to;
};

struct LoopInfo {
    NodeId loop_body{kInvalidNode};
    int64_t max_iterations{0};
    ValueId condition{kInvalidValue};
    std::vector<ValueId> carried_values;
};

struct Subgraph {
    std::vector<NodeId> nodes;
};

struct ControlFlow {
    std::vector<Subgraph> subgraphs;
    std::vector<LoopInfo> loops;
};

class Graph {
public:
    struct DynamicDim {
        ValueId value{kInvalidValue};
        int dim_index{0};
        std::string name;
    };

    std::vector<Node> nodes;
    std::vector<Value> values;
    std::vector<Edge> edges;
    // Constant tensors (ONNX initializers / Constant nodes) keyed by value
    // name. Tensors are held by value; a Tensor copy shares its buffer owner,
    // so the graph keeps the storage alive for the executor.
    std::unordered_map<std::string, Tensor> weights;

    std::vector<DynamicDim> dynamic_dims;
    ControlFlow control_flow;

    bool is_training = false;
    // std::optional<Graph> would be a recursive complete type; shared_ptr keeps
    // Graph copyable while preserving the design's "may have a backward graph".
    std::shared_ptr<Graph> backward_graph;

    Result<BoundGraph> bind(const ShapeMap& shapes) const;
    Result<Graph> build_backward(const std::vector<ValueId>& loss_inputs) const;

    bool is_dynamic() const { return !dynamic_dims.empty(); }
    bool has_control_flow() const { return !control_flow.subgraphs.empty(); }
    bool has_backward() const { return backward_graph != nullptr; }

    const Value* find_value(ValueId id) const;
    const Value* find_value(std::string_view name) const;
    const Value* input(std::string_view name) const;
};

}  // namespace ie
