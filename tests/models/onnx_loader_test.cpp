#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "converter/op_mapping/op_mapping.h"
#include "models/traditional/onnx/onnx_loader.h"

using namespace ie;

namespace {

void put_varint(std::vector<uint8_t>& out, uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
}

void put_tag(std::vector<uint8_t>& out, uint32_t field, uint32_t wire) {
    put_varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

void put_varint_field(std::vector<uint8_t>& out, uint32_t field,
                      uint64_t value) {
    put_tag(out, field, 0);
    put_varint(out, value);
}

void put_bytes_field(std::vector<uint8_t>& out, uint32_t field,
                     const std::vector<uint8_t>& bytes) {
    put_tag(out, field, 2);
    put_varint(out, bytes.size());
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void put_string_field(std::vector<uint8_t>& out, uint32_t field,
                      const std::string& value) {
    put_tag(out, field, 2);
    put_varint(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

void put_packed_float_field(std::vector<uint8_t>& out, uint32_t field,
                            const std::vector<float>& values) {
    std::vector<uint8_t> bytes;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    put_bytes_field(out, field, bytes);
}

std::vector<uint8_t> model_with_initializer(const std::vector<uint8_t>& tensor) {
    std::vector<uint8_t> graph;
    put_bytes_field(graph, 5, tensor);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);
    return model;
}

}  // namespace

TEST(OnnxLoaderTest, ParsesTinyAddAndPreservesWeight) {
    auto graph = load_onnx_model(std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx");
    ASSERT_TRUE(graph.has_value()) << graph.error().message;
    ASSERT_EQ(graph->nodes.size(), 1u);
    EXPECT_EQ(graph->nodes[0].op, OpType::Add);
    ASSERT_EQ(graph->nodes[0].inputs.size(), 2u);
    ASSERT_EQ(graph->nodes[0].outputs.size(), 1u);
    EXPECT_NE(graph->find_value("x"), nullptr);
    EXPECT_NE(graph->find_value("w"), nullptr);
    EXPECT_NE(graph->find_value("y"), nullptr);
    ASSERT_EQ(graph->weights.count("w"), 1u);
    const Tensor& w = graph->weights.at("w");
    EXPECT_TRUE(w.shape == (Shape{{2, 2}}));
    EXPECT_EQ(w.dtype, DType::F32);
    EXPECT_FLOAT_EQ(w.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(w.data<float>()[3], 4.0f);
    EXPECT_TRUE(graph->find_value("y")->is_output);
}

TEST(OnnxLoaderTest, RejectsTruncatedFileAndMissingFile) {
    const std::string path = std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx";
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input.good());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    ASSERT_GT(bytes.size(), 8u);
    bytes.resize(8);
    auto parsed = parse_onnx_model(bytes);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().code, Error::Code::InvalidArgument);

    EXPECT_EQ(load_onnx_model(std::string(IE_TEST_DATA_DIR) +
                              "/does_not_exist.onnx").error().code,
              Error::Code::InvalidArgument);
}

TEST(OnnxLoaderTest, RejectsMalformedWireInput) {
    EXPECT_EQ(parse_onnx_model(std::span<const uint8_t>{}).error().code,
              Error::Code::InvalidArgument);
    // Field 1, wire type 3 (start-group) is not part of the supported subset.
    const std::vector<uint8_t> group_wire{0x0b};
    EXPECT_EQ(parse_onnx_model(group_wire).error().code,
              Error::Code::InvalidArgument);
}

TEST(OnnxLoaderTest, RejectsUnknownTensorDtype) {
    std::vector<uint8_t> tensor;
    put_varint_field(tensor, 2, 99);
    const auto model = model_with_initializer(tensor);
    auto parsed = parse_onnx_model(model);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().code, Error::Code::InvalidArgument);
}

TEST(OnnxLoaderTest, ParsesPackedAttributeLists) {
    std::vector<uint8_t> strides;
    put_varint(strides, 1);
    put_varint(strides, 2);

    std::vector<uint8_t> strides_attr;
    put_string_field(strides_attr, 1, "strides");
    put_bytes_field(strides_attr, 8, strides);

    const float alpha = 0.5f;
    std::vector<uint8_t> alpha_attr;
    put_string_field(alpha_attr, 1, "alpha");
    put_tag(alpha_attr, 2, 5);
    uint32_t bits = 0;
    std::memcpy(&bits, &alpha, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
        alpha_attr.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }

    std::vector<uint8_t> node;
    put_string_field(node, 4, "Conv");
    put_bytes_field(node, 5, strides_attr);
    put_bytes_field(node, 5, alpha_attr);

    std::vector<uint8_t> graph;
    put_bytes_field(graph, 1, node);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);

    auto parsed = parse_onnx_model(model);
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    ASSERT_EQ(parsed->graph.nodes.size(), 1u);
    const OnnxNode& parsed_node = parsed->graph.nodes[0];
    EXPECT_EQ(parsed_node.op_type, "Conv");
    ASSERT_EQ(parsed_node.attributes.size(), 2u);
    const auto* strides_value =
        std::get_if<std::vector<int64_t>>(&parsed_node.attributes[0].value);
    ASSERT_NE(strides_value, nullptr);
    EXPECT_EQ(*strides_value, (std::vector<int64_t>{1, 2}));
    const auto* alpha_value =
        std::get_if<double>(&parsed_node.attributes[1].value);
    ASSERT_NE(alpha_value, nullptr);
    EXPECT_DOUBLE_EQ(*alpha_value, 0.5);
}

TEST(OnnxLoaderTest, FoldsConstantNodesAndConvertsInt64Weights) {
    std::vector<uint8_t> int64_dims;
    put_varint(int64_dims, 2);
    std::vector<uint8_t> int64_values;
    put_varint(int64_values, 3);
    put_varint(int64_values, 4);
    std::vector<uint8_t> int64_tensor;
    put_bytes_field(int64_tensor, 1, int64_dims);
    put_varint_field(int64_tensor, 2, 7);
    put_bytes_field(int64_tensor, 7, int64_values);
    put_string_field(int64_tensor, 8, "shape");

    std::vector<uint8_t> constant_dims;
    put_varint(constant_dims, 1);
    std::vector<uint8_t> constant_tensor;
    put_bytes_field(constant_tensor, 1, constant_dims);
    put_varint_field(constant_tensor, 2, 1);
    put_packed_float_field(constant_tensor, 4, {2.0f});

    std::vector<uint8_t> value_attr;
    put_string_field(value_attr, 1, "value");
    put_bytes_field(value_attr, 5, constant_tensor);

    std::vector<uint8_t> constant_node;
    put_string_field(constant_node, 2, "scale");
    put_string_field(constant_node, 4, "Constant");
    put_bytes_field(constant_node, 5, value_attr);

    std::vector<uint8_t> graph_bytes;
    put_bytes_field(graph_bytes, 1, constant_node);
    put_bytes_field(graph_bytes, 5, int64_tensor);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph_bytes);

    const auto path = std::filesystem::temp_directory_path() /
                      "ie_onnx_loader_constant_int64_test.onnx";
    {
        std::ofstream output(path, std::ios::binary);
        ASSERT_TRUE(output.good());
        output.write(reinterpret_cast<const char*>(model.data()),
                     static_cast<std::streamsize>(model.size()));
    }
    auto graph = load_onnx_model(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(graph.has_value()) << graph.error().message;
    EXPECT_TRUE(graph->nodes.empty());
    ASSERT_EQ(graph->weights.count("shape"), 1u);
    ASSERT_EQ(graph->weights.count("scale"), 1u);
    const Tensor& shape = graph->weights.at("shape");
    EXPECT_TRUE(shape.shape == (Shape{{2}}));
    EXPECT_FLOAT_EQ(shape.data<float>()[0], 3.0f);
    EXPECT_FLOAT_EQ(shape.data<float>()[1], 4.0f);
    EXPECT_FALSE(graph->find_value("shape")->is_input);
    const Tensor& scale = graph->weights.at("scale");
    EXPECT_FLOAT_EQ(scale.data<float>()[0], 2.0f);
}

TEST(OnnxMappingTest, MapsTraditionalWhitelistAndRejectsUnknown) {
    OnnxMapping mapping;
    EXPECT_EQ(*mapping.map("Conv"), OpType::Conv2D);
    EXPECT_EQ(*mapping.map("GlobalAveragePool"), OpType::GlobalAvgPool);
    EXPECT_EQ(*mapping.map("Flatten"), OpType::Flatten);
    EXPECT_EQ(mapping.map("DefinitelyNotAnOp").error().code,
              Error::Code::Unsupported);
}
