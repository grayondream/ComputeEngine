#include "models/traditional/onnx/onnx_wire.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ie {
namespace {

// Highest TensorProto.DataType value known to the ONNX spec (FLOAT6E3M2).
constexpr int32_t kMaxOnnxDtype = 28;

class ProtoReader {
public:
    explicit ProtoReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    bool empty() const { return pos_ >= bytes_.size(); }

    Result<uint64_t> read_varint() {
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (pos_ >= bytes_.size()) {
                return Error::invalid_argument("onnx: truncated varint");
            }
            const uint8_t byte = bytes_[pos_++];
            value |= static_cast<uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) {
                if (shift == 63 && (byte & 0x7e) != 0) {
                    return Error::invalid_argument(
                        "onnx: varint exceeds 64 bits");
                }
                return value;
            }
        }
        return Error::invalid_argument("onnx: varint exceeds 64 bits");
    }

    // Returns {field_number, wire_type}.
    Result<std::pair<uint32_t, uint32_t>> read_tag() {
        auto tag = read_varint();
        if (!tag) {
            return tag.error();
        }
        const uint32_t field = static_cast<uint32_t>(*tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(*tag & 0x7);
        if (field == 0) {
            return Error::invalid_argument("onnx: field number zero");
        }
        if (wire != 0 && wire != 1 && wire != 2 && wire != 5) {
            return Error::invalid_argument("onnx: unsupported wire type " +
                                           std::to_string(wire));
        }
        return std::make_pair(field, wire);
    }

    Result<std::span<const uint8_t>> read_length_delimited() {
        auto length = read_varint();
        if (!length) {
            return length.error();
        }
        const uint64_t remaining =
            static_cast<uint64_t>(bytes_.size() - pos_);
        if (*length > remaining) {
            return Error::invalid_argument(
                "onnx: truncated length-delimited field");
        }
        const auto size = static_cast<std::size_t>(*length);
        auto sub = bytes_.subspan(pos_, size);
        pos_ += size;
        return sub;
    }

    Result<std::span<const uint8_t>> enter_message() {
        return read_length_delimited();
    }

    Result<uint32_t> read_fixed32() {
        if (bytes_.size() - pos_ < 4) {
            return Error::invalid_argument("onnx: truncated fixed32 field");
        }
        const uint32_t value =
            static_cast<uint32_t>(bytes_[pos_]) |
            (static_cast<uint32_t>(bytes_[pos_ + 1]) << 8) |
            (static_cast<uint32_t>(bytes_[pos_ + 2]) << 16) |
            (static_cast<uint32_t>(bytes_[pos_ + 3]) << 24);
        pos_ += 4;
        return value;
    }

    Result<uint64_t> read_fixed64() {
        if (bytes_.size() - pos_ < 8) {
            return Error::invalid_argument("onnx: truncated fixed64 field");
        }
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<uint64_t>(bytes_[pos_ + static_cast<std::size_t>(i)])
                     << (8 * i);
        }
        pos_ += 8;
        return value;
    }

    Result<std::string> read_string() {
        auto bytes = read_length_delimited();
        if (!bytes) {
            return bytes.error();
        }
        if (bytes->empty()) {
            return std::string();
        }
        return std::string(reinterpret_cast<const char*>(bytes->data()),
                           bytes->size());
    }

    Status skip(uint32_t wire_type) {
        switch (wire_type) {
            case 0: {
                auto value = read_varint();
                if (!value) {
                    return value.error();
                }
                return {};
            }
            case 1:
                return skip_bytes(8);
            case 2: {
                auto bytes = read_length_delimited();
                if (!bytes) {
                    return bytes.error();
                }
                return {};
            }
            case 5:
                return skip_bytes(4);
            default:
                return Error::invalid_argument("onnx: unsupported wire type " +
                                               std::to_string(wire_type));
        }
    }

private:
    Status skip_bytes(std::size_t count) {
        if (bytes_.size() - pos_ < count) {
            return Error::invalid_argument("onnx: truncated fixed-width field");
        }
        pos_ += count;
        return {};
    }

    std::span<const uint8_t> bytes_;
    std::size_t pos_{0};
};

Status read_repeated_int64(ProtoReader& reader, uint32_t wire,
                           std::vector<int64_t>& out) {
    if (wire == 0) {
        auto value = reader.read_varint();
        if (!value) {
            return value.error();
        }
        out.push_back(static_cast<int64_t>(*value));
        return {};
    }
    if (wire != 2) {
        return Error::invalid_argument("onnx: unexpected wire type for int64");
    }
    auto packed = reader.read_length_delimited();
    if (!packed) {
        return packed.error();
    }
    ProtoReader sub(*packed);
    while (!sub.empty()) {
        auto value = sub.read_varint();
        if (!value) {
            return value.error();
        }
        out.push_back(static_cast<int64_t>(*value));
    }
    return {};
}

Status read_repeated_int32(ProtoReader& reader, uint32_t wire,
                           std::vector<int32_t>& out) {
    if (wire == 0) {
        auto value = reader.read_varint();
        if (!value) {
            return value.error();
        }
        out.push_back(static_cast<int32_t>(*value));
        return {};
    }
    if (wire != 2) {
        return Error::invalid_argument("onnx: unexpected wire type for int32");
    }
    auto packed = reader.read_length_delimited();
    if (!packed) {
        return packed.error();
    }
    ProtoReader sub(*packed);
    while (!sub.empty()) {
        auto value = sub.read_varint();
        if (!value) {
            return value.error();
        }
        out.push_back(static_cast<int32_t>(*value));
    }
    return {};
}

Status read_repeated_float(ProtoReader& reader, uint32_t wire,
                           std::vector<float>& out) {
    if (wire == 5) {
        auto value = reader.read_fixed32();
        if (!value) {
            return value.error();
        }
        out.push_back(std::bit_cast<float>(*value));
        return {};
    }
    if (wire != 2) {
        return Error::invalid_argument("onnx: unexpected wire type for float");
    }
    auto packed = reader.read_length_delimited();
    if (!packed) {
        return packed.error();
    }
    if (packed->size() % 4 != 0) {
        return Error::invalid_argument("onnx: packed float length mismatch");
    }
    ProtoReader sub(*packed);
    while (!sub.empty()) {
        auto value = sub.read_fixed32();
        if (!value) {
            return value.error();
        }
        out.push_back(std::bit_cast<float>(*value));
    }
    return {};
}

Status read_repeated_double(ProtoReader& reader, uint32_t wire,
                            std::vector<double>& out) {
    if (wire == 1) {
        auto value = reader.read_fixed64();
        if (!value) {
            return value.error();
        }
        out.push_back(std::bit_cast<double>(*value));
        return {};
    }
    if (wire != 2) {
        return Error::invalid_argument("onnx: unexpected wire type for double");
    }
    auto packed = reader.read_length_delimited();
    if (!packed) {
        return packed.error();
    }
    if (packed->size() % 8 != 0) {
        return Error::invalid_argument("onnx: packed double length mismatch");
    }
    ProtoReader sub(*packed);
    while (!sub.empty()) {
        auto value = sub.read_fixed64();
        if (!value) {
            return value.error();
        }
        out.push_back(std::bit_cast<double>(*value));
    }
    return {};
}

Result<OnnxTensor> parse_tensor(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxTensor tensor;
    bool external = false;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1:
                if (auto status =
                        read_repeated_int64(reader, wire, tensor.dims);
                    !status) {
                    return status.error();
                }
                break;
            case 2: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor data_type is not a varint");
                }
                auto value = reader.read_varint();
                if (!value) {
                    return value.error();
                }
                if (*value >
                    static_cast<uint64_t>(
                        std::numeric_limits<uint32_t>::max())) {
                    return Error::invalid_argument(
                        "onnx: tensor data_type is out of range");
                }
                tensor.data_type = static_cast<int32_t>(*value);
                break;
            }
            case 4:
                if (auto status =
                        read_repeated_float(reader, wire, tensor.float_data);
                    !status) {
                    return status.error();
                }
                break;
            case 5:
                if (auto status =
                        read_repeated_int32(reader, wire, tensor.int32_data);
                    !status) {
                    return status.error();
                }
                break;
            case 7:
                if (auto status =
                        read_repeated_int64(reader, wire, tensor.int64_data);
                    !status) {
                    return status.error();
                }
                break;
            case 8: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: tensor name is not length-delimited");
                }
                auto name = reader.read_string();
                if (!name) {
                    return name.error();
                }
                tensor.name = std::move(*name);
                break;
            }
            case 9: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: tensor raw_data is not length-delimited");
                }
                auto raw = reader.read_length_delimited();
                if (!raw) {
                    return raw.error();
                }
                tensor.raw_data.assign(raw->begin(), raw->end());
                break;
            }
            case 10:
                if (auto status = read_repeated_double(reader, wire,
                                                       tensor.double_data);
                    !status) {
                    return status.error();
                }
                break;
            case 11:
                // uint64_data is outside the supported F32 conversion subset.
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
            case 13:
                external = true;
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
            case 14: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: tensor data_location is not a varint");
                }
                auto location = reader.read_varint();
                if (!location) {
                    return location.error();
                }
                if (*location != 0) {
                    external = true;
                }
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }

    if (external) {
        return Error::unsupported("onnx: external tensor data is not supported");
    }
    if (tensor.data_type == 0) {
        return Error::invalid_argument("onnx: tensor dtype is undefined");
    }
    if (tensor.data_type < 0 || tensor.data_type > kMaxOnnxDtype) {
        return Error::invalid_argument(
            "onnx: unknown tensor dtype " + std::to_string(tensor.data_type));
    }
    return tensor;
}

Result<OnnxAttribute> parse_attribute(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxAttribute attribute;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: attribute name is not length-delimited");
                }
                auto name = reader.read_string();
                if (!name) {
                    return name.error();
                }
                attribute.name = std::move(*name);
                break;
            }
            case 2: {
                if (wire != 5) {
                    return Error::invalid_argument(
                        "onnx: attribute f is not a fixed32");
                }
                auto value = reader.read_fixed32();
                if (!value) {
                    return value.error();
                }
                attribute.value = static_cast<double>(std::bit_cast<float>(*value));
                attribute.has_value = true;
                break;
            }
            case 3: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: attribute i is not a varint");
                }
                auto value = reader.read_varint();
                if (!value) {
                    return value.error();
                }
                attribute.value = static_cast<int64_t>(*value);
                attribute.has_value = true;
                break;
            }
            case 4: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: attribute s is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                attribute.value = std::move(*value);
                attribute.has_value = true;
                break;
            }
            case 5: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: attribute t is not length-delimited");
                }
                auto tensor_bytes = reader.enter_message();
                if (!tensor_bytes) {
                    return tensor_bytes.error();
                }
                auto tensor = parse_tensor(*tensor_bytes);
                if (!tensor) {
                    return tensor.error();
                }
                attribute.value = std::move(*tensor);
                attribute.has_value = true;
                break;
            }
            case 7: {
                std::vector<float> floats;
                if (auto status = read_repeated_float(reader, wire, floats);
                    !status) {
                    return status.error();
                }
                auto* existing =
                    std::get_if<std::vector<double>>(&attribute.value);
                if (existing == nullptr) {
                    attribute.value = std::vector<double>{};
                    existing =
                        std::get_if<std::vector<double>>(&attribute.value);
                }
                existing->insert(existing->end(), floats.begin(),
                                 floats.end());
                attribute.has_value = true;
                break;
            }
            case 8: {
                std::vector<int64_t> values;
                if (auto status = read_repeated_int64(reader, wire, values);
                    !status) {
                    return status.error();
                }
                auto* existing =
                    std::get_if<std::vector<int64_t>>(&attribute.value);
                if (existing == nullptr) {
                    attribute.value = std::vector<int64_t>{};
                    existing =
                        std::get_if<std::vector<int64_t>>(&attribute.value);
                }
                existing->insert(existing->end(), values.begin(),
                                 values.end());
                attribute.has_value = true;
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }
    return attribute;
}

Result<OnnxNode> parse_node(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxNode node;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: node input is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                node.inputs.push_back(std::move(*value));
                break;
            }
            case 2: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: node output is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                node.outputs.push_back(std::move(*value));
                break;
            }
            case 3: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: node name is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                node.name = std::move(*value);
                break;
            }
            case 4: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: node op_type is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                node.op_type = std::move(*value);
                break;
            }
            case 5: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: node attribute is not length-delimited");
                }
                auto attribute_bytes = reader.enter_message();
                if (!attribute_bytes) {
                    return attribute_bytes.error();
                }
                auto attribute = parse_attribute(*attribute_bytes);
                if (!attribute) {
                    return attribute.error();
                }
                node.attributes.push_back(std::move(*attribute));
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }
    return node;
}

Result<int64_t> parse_dimension(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    std::optional<int64_t> dim_value;
    bool has_dim_param = false;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: dim_value is not a varint");
                }
                auto value = reader.read_varint();
                if (!value) {
                    return value.error();
                }
                dim_value = static_cast<int64_t>(*value);
                break;
            }
            case 2: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: dim_param is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                has_dim_param = true;
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }

    if (has_dim_param || !dim_value.has_value()) {
        return int64_t{-1};
    }
    return *dim_value > 0 ? *dim_value : int64_t{-1};
}

Result<std::vector<int64_t>> parse_tensor_shape(
    std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    std::vector<int64_t> dims;
    bool unknown_rank = false;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: shape dimension is not length-delimited");
                }
                auto dim_bytes = reader.enter_message();
                if (!dim_bytes) {
                    return dim_bytes.error();
                }
                auto dim = parse_dimension(*dim_bytes);
                if (!dim) {
                    return dim.error();
                }
                dims.push_back(*dim);
                break;
            }
            case 2: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: shape unknown_rank is not a varint");
                }
                auto value = reader.read_varint();
                if (!value) {
                    return value.error();
                }
                unknown_rank = *value != 0;
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }

    if (unknown_rank) {
        dims.clear();
    }
    return dims;
}

Status parse_tensor_type(std::span<const uint8_t> bytes,
                         OnnxValueInfo& info) {
    ProtoReader reader(bytes);
    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 0) {
                    return Error::invalid_argument(
                        "onnx: elem_type is not a varint");
                }
                auto value = reader.read_varint();
                if (!value) {
                    return value.error();
                }
                info.elem_type = static_cast<int32_t>(*value);
                break;
            }
            case 2: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: tensor shape is not length-delimited");
                }
                auto shape_bytes = reader.enter_message();
                if (!shape_bytes) {
                    return shape_bytes.error();
                }
                auto dims = parse_tensor_shape(*shape_bytes);
                if (!dims) {
                    return dims.error();
                }
                info.dims = std::move(*dims);
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }
    return {};
}

Status parse_type(std::span<const uint8_t> bytes, OnnxValueInfo& info) {
    ProtoReader reader(bytes);
    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        if (field == 1) {
            if (wire != 2) {
                return Error::invalid_argument(
                    "onnx: tensor_type is not length-delimited");
            }
            auto tensor_type = reader.enter_message();
            if (!tensor_type) {
                return tensor_type.error();
            }
            if (auto status = parse_tensor_type(*tensor_type, info); !status) {
                return status.error();
            }
        } else if (auto status = reader.skip(wire); !status) {
            return status.error();
        }
    }
    return {};
}

Result<OnnxValueInfo> parse_value_info(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxValueInfo info;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: value name is not length-delimited");
                }
                auto value = reader.read_string();
                if (!value) {
                    return value.error();
                }
                info.name = std::move(*value);
                break;
            }
            case 2: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: value type is not length-delimited");
                }
                auto type_bytes = reader.enter_message();
                if (!type_bytes) {
                    return type_bytes.error();
                }
                if (auto status = parse_type(*type_bytes, info); !status) {
                    return status.error();
                }
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }
    return info;
}

Result<OnnxGraph> parse_graph(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxGraph graph;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        switch (field) {
            case 1:
            case 5:
            case 11:
            case 12:
            case 13: {
                if (wire != 2) {
                    return Error::invalid_argument(
                        "onnx: graph message field is not length-delimited");
                }
                auto message = reader.enter_message();
                if (!message) {
                    return message.error();
                }
                if (field == 1) {
                    auto node = parse_node(*message);
                    if (!node) {
                        return node.error();
                    }
                    graph.nodes.push_back(std::move(*node));
                } else if (field == 5) {
                    auto tensor = parse_tensor(*message);
                    if (!tensor) {
                        return tensor.error();
                    }
                    graph.initializers.push_back(std::move(*tensor));
                } else {
                    auto info = parse_value_info(*message);
                    if (!info) {
                        return info.error();
                    }
                    if (field == 11) {
                        graph.inputs.push_back(std::move(*info));
                    } else if (field == 12) {
                        graph.outputs.push_back(std::move(*info));
                    } else {
                        graph.value_info.push_back(std::move(*info));
                    }
                }
                break;
            }
            default:
                if (auto status = reader.skip(wire); !status) {
                    return status.error();
                }
                break;
        }
    }
    return graph;
}

Result<OnnxModel> parse_model(std::span<const uint8_t> bytes) {
    ProtoReader reader(bytes);
    OnnxModel model;
    bool saw_graph = false;

    while (!reader.empty()) {
        auto tag = reader.read_tag();
        if (!tag) {
            return tag.error();
        }
        const auto [field, wire] = *tag;
        if (field == 7) {
            if (wire != 2) {
                return Error::invalid_argument(
                    "onnx: model graph is not length-delimited");
            }
            auto graph_bytes = reader.enter_message();
            if (!graph_bytes) {
                return graph_bytes.error();
            }
            auto graph = parse_graph(*graph_bytes);
            if (!graph) {
                return graph.error();
            }
            model.graph = std::move(*graph);
            saw_graph = true;
        } else if (auto status = reader.skip(wire); !status) {
            return status.error();
        }
    }

    if (!saw_graph) {
        return Error::invalid_argument("onnx: model has no graph");
    }
    return model;
}

}  // namespace

Result<OnnxModel> parse_onnx_model(std::span<const uint8_t> bytes) {
    return parse_model(bytes);
}

}  // namespace ie
