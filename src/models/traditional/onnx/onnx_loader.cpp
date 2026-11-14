#include "models/traditional/onnx/onnx_loader.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "converter/op_mapping/op_mapping.h"
#include "core/memory/malloc_allocator.h"

namespace ie {
namespace {

constexpr int32_t kOnnxFloat = 1;
constexpr int32_t kOnnxInt32 = 6;
constexpr int32_t kOnnxInt64 = 7;
constexpr int32_t kOnnxDouble = 11;

float read_le_f32(const uint8_t* bytes) {
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    return std::bit_cast<float>(bits);
}

double read_le_f64(const uint8_t* bytes) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return std::bit_cast<double>(bits);
}

int32_t read_le_i32(const uint8_t* bytes) {
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    return static_cast<int32_t>(bits);
}

int64_t read_le_i64(const uint8_t* bytes) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return static_cast<int64_t>(bits);
}

Result<std::size_t> element_count(const OnnxTensor& tensor) {
    if (tensor.dims.empty()) {
        return std::size_t{1};
    }
    uint64_t count = 1;
    for (int64_t dim : tensor.dims) {
        if (dim <= 0) {
            return Error::invalid_argument("onnx: tensor '" + tensor.name +
                                           "' has a non-positive dimension");
        }
        const auto positive = static_cast<uint64_t>(dim);
        if (count > std::numeric_limits<uint64_t>::max() / positive) {
            return Error::invalid_argument("onnx: tensor '" + tensor.name +
                                           "' element count overflows");
        }
        count *= positive;
    }
    if (count > std::numeric_limits<std::size_t>::max()) {
        return Error::invalid_argument("onnx: tensor '" + tensor.name +
                                       "' is too large");
    }
    return static_cast<std::size_t>(count);
}

Result<Shape> tensor_shape(const OnnxTensor& tensor) {
    auto count = element_count(tensor);
    if (!count) {
        return count.error();
    }
    if (tensor.dims.empty()) {
        return Shape{{1}};
    }
    return Shape{tensor.dims};
}

Result<std::vector<float>> tensor_values(const OnnxTensor& tensor) {
    auto count = element_count(tensor);
    if (!count) {
        return count.error();
    }
    const std::size_t n = *count;

    std::vector<float> values;
    switch (tensor.data_type) {
        case kOnnxFloat:
            if (!tensor.raw_data.empty()) {
                if (tensor.raw_data.size() / 4 != n ||
                    tensor.raw_data.size() % 4 != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' raw float payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = read_le_f32(tensor.raw_data.data() + i * 4);
                }
            } else {
                if (tensor.float_data.size() != n) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' float payload size mismatch");
                }
                values = tensor.float_data;
            }
            break;
        case kOnnxInt64:
            if (!tensor.raw_data.empty()) {
                if (tensor.raw_data.size() / 8 != n ||
                    tensor.raw_data.size() % 8 != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' raw int64 payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(
                        read_le_i64(tensor.raw_data.data() + i * 8));
                }
            } else {
                if (tensor.int64_data.size() != n) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' int64 payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(tensor.int64_data[i]);
                }
            }
            break;
        case kOnnxInt32:
            if (!tensor.raw_data.empty()) {
                if (tensor.raw_data.size() / 4 != n ||
                    tensor.raw_data.size() % 4 != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' raw int32 payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(
                        read_le_i32(tensor.raw_data.data() + i * 4));
                }
            } else {
                if (tensor.int32_data.size() != n) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' int32 payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(tensor.int32_data[i]);
                }
            }
            break;
        case kOnnxDouble:
            if (!tensor.raw_data.empty()) {
                if (tensor.raw_data.size() / 8 != n ||
                    tensor.raw_data.size() % 8 != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' raw double payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(
                        read_le_f64(tensor.raw_data.data() + i * 8));
                }
            } else {
                if (tensor.double_data.size() != n) {
                    return Error::invalid_argument(
                        "onnx: tensor '" + tensor.name +
                        "' double payload size mismatch");
                }
                values.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    values[i] = static_cast<float>(tensor.double_data[i]);
                }
            }
            break;
        default:
            return Error::unsupported("onnx: tensor dtype " +
                                      std::to_string(tensor.data_type) +
                                      " is not supported");
    }
    return values;
}

Shape shape_from_value_info(const OnnxValueInfo& info) {
    Shape shape;
    shape.dims.reserve(info.dims.size());
    for (int64_t dim : info.dims) {
        shape.dims.push_back(dim > 0 ? dim : int64_t{-1});
    }
    return shape;
}

const OnnxTensor* constant_tensor(const OnnxNode& node) {
    for (const OnnxAttribute& attribute : node.attributes) {
        if (attribute.name == "value" && attribute.has_value) {
            if (const auto* tensor =
                    std::get_if<OnnxTensor>(&attribute.value)) {
                return tensor;
            }
        }
    }
    return nullptr;
}

class OnnxGraphBuilder {
public:
    Result<Graph> build(const OnnxGraph& source) {
        std::unordered_set<std::string> initializer_names;
        for (const OnnxTensor& initializer : source.initializers) {
            if (initializer.name.empty()) {
                return Error::invalid_argument("onnx: initializer without name");
            }
            initializer_names.insert(initializer.name);
        }

        for (const OnnxValueInfo& info : source.inputs) {
            if (info.name.empty()) {
                continue;
            }
            const ValueId id = ensure_value(info.name);
            if (initializer_names.count(info.name) == 0) {
                graph_.values[id].is_input = true;
            }
            apply_shape(id, info);
        }
        for (const OnnxValueInfo& info : source.outputs) {
            if (info.name.empty()) {
                continue;
            }
            const ValueId id = ensure_value(info.name);
            graph_.values[id].is_output = true;
            apply_shape(id, info);
        }
        for (const OnnxValueInfo& info : source.value_info) {
            if (info.name.empty()) {
                continue;
            }
            const ValueId id = ensure_value(info.name);
            apply_shape(id, info);
        }
        for (const OnnxTensor& initializer : source.initializers) {
            if (auto status = add_weight(initializer.name, initializer);
                !status) {
                return status.error();
            }
        }
        for (const OnnxNode& source_node : source.nodes) {
            if (auto status = add_node(source_node); !status) {
                return status.error();
            }
        }
        return std::move(graph_);
    }

private:
    ValueId ensure_value(const std::string& name) {
        const auto existing = ids_.find(name);
        if (existing != ids_.end()) {
            return existing->second;
        }
        Value value;
        value.id = static_cast<ValueId>(graph_.values.size());
        value.name = name;
        graph_.values.push_back(std::move(value));
        const ValueId id = graph_.values.back().id;
        ids_.emplace(name, id);
        return id;
    }

    void apply_shape(ValueId id, const OnnxValueInfo& info) {
        if (info.dims.empty()) {
            return;
        }
        graph_.values[id].shape = shape_from_value_info(info);
    }

    Status add_weight(const std::string& name, const OnnxTensor& tensor) {
        auto values = tensor_values(tensor);
        if (!values) {
            return values.error();
        }
        auto shape = tensor_shape(tensor);
        if (!shape) {
            return shape.error();
        }
        auto allocated = Tensor::allocate(*shape, DType::F32, allocator_,
                                          MemoryKind::WEIGHT_CONST);
        if (!allocated) {
            return allocated.error();
        }
        std::copy(values->begin(), values->end(), allocated->data<float>());

        const ValueId id = ensure_value(name);
        Value& value = graph_.values[id];
        value.shape = *shape;
        value.dtype = DType::F32;
        value.is_input = false;
        graph_.weights[name] = std::move(*allocated);
        return {};
    }

    Status fold_constant(const OnnxNode& node) {
        const OnnxTensor* tensor = constant_tensor(node);
        if (tensor == nullptr) {
            return Error::unsupported("onnx: Constant node '" + node.name +
                                      "' without a tensor value is not supported");
        }
        std::string output;
        for (const std::string& name : node.outputs) {
            if (!name.empty()) {
                output = name;
                break;
            }
        }
        if (output.empty()) {
            return Error::invalid_argument(
                "onnx: Constant node without an output name");
        }
        return add_weight(output, *tensor);
    }

    Result<Attrs> convert_attributes(const OnnxNode& node) {
        Attrs attrs;
        for (const OnnxAttribute& attribute : node.attributes) {
            if (!attribute.has_value || attribute.name.empty()) {
                continue;
            }
            if (const auto* value = std::get_if<int64_t>(&attribute.value)) {
                attrs[attribute.name] = *value;
            } else if (const auto* value =
                           std::get_if<double>(&attribute.value)) {
                attrs[attribute.name] = *value;
            } else if (const auto* value =
                           std::get_if<std::string>(&attribute.value)) {
                attrs[attribute.name] = *value;
            } else if (const auto* value =
                           std::get_if<std::vector<int64_t>>(
                               &attribute.value)) {
                attrs[attribute.name] = *value;
            } else if (const auto* value =
                           std::get_if<std::vector<double>>(
                               &attribute.value)) {
                attrs[attribute.name] = *value;
            } else {
                return Error::unsupported(
                    "onnx: tensor attribute '" + attribute.name +
                    "' is only supported on Constant nodes");
            }
        }
        return attrs;
    }

    Status add_node(const OnnxNode& source_node) {
        if (source_node.op_type == "Constant") {
            return fold_constant(source_node);
        }

        auto mapped = mapping_.map(source_node.op_type);
        if (!mapped) {
            return mapped.error();
        }
        auto attrs = convert_attributes(source_node);
        if (!attrs) {
            return attrs.error();
        }
        if (source_node.op_type == "ReduceMean") {
            (*attrs)["reduce_mode"] = std::string("mean");
        } else if (source_node.op_type == "ReduceSum") {
            (*attrs)["reduce_mode"] = std::string("sum");
        }

        Node node;
        node.id = static_cast<NodeId>(graph_.nodes.size());
        node.op = *mapped;
        node.attrs = std::move(*attrs);
        for (const std::string& name : source_node.inputs) {
            if (!name.empty()) {
                node.inputs.push_back(ensure_value(name));
            }
        }
        for (const std::string& name : source_node.outputs) {
            if (!name.empty()) {
                node.outputs.push_back(ensure_value(name));
            }
        }

        graph_.nodes.push_back(std::move(node));
        const NodeId node_id = graph_.nodes.back().id;
        for (ValueId input : graph_.nodes.back().inputs) {
            graph_.edges.push_back(Edge{input, std::nullopt, node_id});
        }
        return {};
    }

    Graph graph_;
    std::unordered_map<std::string, ValueId> ids_;
    std::shared_ptr<MallocAllocator> allocator_ =
        std::make_shared<MallocAllocator>();
    OnnxMapping mapping_;
};

}  // namespace

Result<Graph> load_onnx_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        return Error::invalid_argument("onnx: cannot open model: " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    if (input.bad()) {
        return Error::invalid_argument("onnx: failed to read model: " + path);
    }
    if (bytes.empty()) {
        return Error::invalid_argument("onnx: empty model file: " + path);
    }

    auto model = parse_onnx_model(bytes);
    if (!model) {
        return model.error();
    }
    OnnxGraphBuilder builder;
    return builder.build(model->graph);
}

}  // namespace ie
