#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/shape.h"
#include "compute_engine/core/dtype.h"
#include "compute_engine/core/tensor.h"
#include "compute_engine/op/op.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace compute_engine {
namespace graph {

enum class AttributeType : uint8_t {
    FLOAT, INT, STRING, FLOATS, INTS, SHAPE, TENSOR, GRAPH,
};

struct Attribute {
    std::string name;
    AttributeType type;

    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    Shape shape;
    TensorPtr tensor;

    Attribute() = default;
    Attribute(const std::string& n, float v) : name(n), type(AttributeType::FLOAT), f(v) {}
    Attribute(const std::string& n, int64_t v) : name(n), type(AttributeType::INT), i(v) {}
    Attribute(const std::string& n, const std::string& v) : name(n), type(AttributeType::STRING), s(v) {}
    Attribute(const std::string& n, const std::vector<float>& v) : name(n), type(AttributeType::FLOATS), floats(v) {}
    Attribute(const std::string& n, const std::vector<int64_t>& v) : name(n), type(AttributeType::INTS), ints(v) {}
    Attribute(const std::string& n, const Shape& v) : name(n), type(AttributeType::SHAPE), shape(v) {}
};

struct TensorDef {
    std::string name;
    Shape shape;
    DType dtype = DType::FLOAT32;
    void* data = nullptr;
    bool is_constant = false;
};

using TensorDefPtr = std::shared_ptr<TensorDef>;

struct NodeDef {
    std::string name;
    OpType op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attrs;
    std::string backend_hint;

    const Attribute* GetAttr(const std::string& name) const {
        for (auto& a : attrs)
            if (a.name == name) return &a;
        return nullptr;
    }

    int64_t GetIntAttr(const std::string& name, int64_t default_val = 0) const {
        auto* a = GetAttr(name);
        return (a && a->type == AttributeType::INT) ? a->i : default_val;
    }

    float GetFloatAttr(const std::string& name, float default_val = 0.0f) const {
        auto* a = GetAttr(name);
        return (a && a->type == AttributeType::FLOAT) ? a->f : default_val;
    }
};

struct EdgeDef {
    std::string from_node;
    int from_output = 0;
    std::string to_node;
    int to_input = 0;
    std::string tensor_name;
};

class GraphDef {
public:
    GraphDef() = default;
    explicit GraphDef(const std::string& name) : name_(name) {}

    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    // Node management
    NodeDef* AddNode(const std::string& name, OpType op_type);
    NodeDef* GetNode(const std::string& name);
    const NodeDef* GetNode(const std::string& name) const;
    size_t num_nodes() const { return nodes_.size(); }
    const std::vector<std::unique_ptr<NodeDef>>& nodes() const { return nodes_; }

    // Tensor management
    void AddInput(const std::string& name, const Shape& shape, DType dtype = DType::FLOAT32);
    void AddOutput(const std::string& name);
    void SetConstant(const std::string& name, TensorPtr tensor);

    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    // Tensor registry
    TensorDefPtr GetTensor(const std::string& name);
    void RegisterTensor(TensorDefPtr tensor);
    const std::unordered_map<std::string, TensorDefPtr>& tensors() const { return tensors_; }

    // Topological sort
    Status TopologicalSort(std::vector<NodeDef*>& sorted) const;

    // Validation
    Status Validate() const;

    // Shape inference for all nodes
    Status InferShapes();

    // Dot graph export for visualization
    std::string ToDot() const;

    std::string ToString() const;

private:
    std::string name_;
    std::vector<std::unique_ptr<NodeDef>> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, TensorDefPtr> tensors_;
    std::unordered_map<std::string, size_t> node_index_;
};

} // namespace graph
} // namespace compute_engine
