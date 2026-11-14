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

void put_packed_double_field(std::vector<uint8_t>& out, uint32_t field,
                             const std::vector<double>& values) {
    std::vector<uint8_t> bytes;
    for (double value : values) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    put_bytes_field(out, field, bytes);
}

void put_packed_varint_field(std::vector<uint8_t>& out, uint32_t field,
                             const std::vector<uint64_t>& values) {
    std::vector<uint8_t> bytes;
    for (uint64_t value : values) {
        put_varint(bytes, value);
    }
    put_bytes_field(out, field, bytes);
}

void put_fixed32_field(std::vector<uint8_t>& out, uint32_t field,
                       float value) {
    put_tag(out, field, 5);
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }
}

std::vector<uint8_t> encode_f32(const std::vector<float>& values) {
    std::vector<uint8_t> bytes;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    return bytes;
}

std::vector<uint8_t> encode_i32(const std::vector<int32_t>& values) {
    std::vector<uint8_t> bytes;
    for (int32_t value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    return bytes;
}

std::vector<uint8_t> encode_i64(const std::vector<int64_t>& values) {
    std::vector<uint8_t> bytes;
    for (int64_t value : values) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    return bytes;
}

std::vector<uint8_t> encode_f64(const std::vector<double>& values) {
    std::vector<uint8_t> bytes;
    for (double value : values) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
    return bytes;
}

std::vector<uint8_t> make_raw_tensor(const std::string& name, int64_t dim,
                                     uint32_t data_type,
                                     const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> dims;
    put_varint(dims, static_cast<uint64_t>(dim));
    std::vector<uint8_t> tensor;
    put_bytes_field(tensor, 1, dims);
    put_varint_field(tensor, 2, data_type);
    put_bytes_field(tensor, 9, raw);
    put_string_field(tensor, 8, name);
    return tensor;
}

std::vector<uint8_t> make_model_with_initializers(
    const std::vector<std::vector<uint8_t>>& initializers) {
    std::vector<uint8_t> graph;
    for (const auto& initializer : initializers) {
        put_bytes_field(graph, 5, initializer);
    }
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);
    return model;
}

Result<Graph> load_wire_model(const std::vector<uint8_t>& model,
                              const std::string& stem) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ie_onnx_" + stem + ".onnx");
    {
        std::ofstream output(path, std::ios::binary);
        if (!output.good()) {
            return Error::invalid_argument("failed to write temp onnx fixture");
        }
        output.write(reinterpret_cast<const char*>(model.data()),
                     static_cast<std::streamsize>(model.size()));
    }
    auto graph = load_onnx_model(path.string());
    std::filesystem::remove(path);
    return graph;
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

    std::vector<uint8_t> wide_dtype_tensor;
    put_varint_field(wide_dtype_tensor, 2, (uint64_t{1} << 32) | 1);
    auto wide_dtype =
        parse_onnx_model(model_with_initializer(wide_dtype_tensor));
    ASSERT_FALSE(wide_dtype.has_value());
    EXPECT_EQ(wide_dtype.error().code, Error::Code::InvalidArgument);
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

TEST(OnnxLoaderTest, DecodesRawAndTypedTensorPayloads) {
    const auto raw_f32 =
        make_raw_tensor("raw_f32", 2, 1, encode_f32({1.5f, -2.5f}));
    const auto raw_i32 = make_raw_tensor("raw_i32", 2, 6, encode_i32({7, -8}));
    const auto raw_i64 = make_raw_tensor("raw_i64", 1, 7, encode_i64({9}));
    const auto raw_f64 = make_raw_tensor("raw_f64", 1, 11, encode_f64({0.25}));

    std::vector<uint8_t> typed_i32_dims;
    put_varint(typed_i32_dims, 2);
    std::vector<uint8_t> typed_i32;
    put_bytes_field(typed_i32, 1, typed_i32_dims);
    put_varint_field(typed_i32, 2, 6);
    put_packed_varint_field(typed_i32, 5, {4, 5});
    put_string_field(typed_i32, 8, "typed_i32");

    std::vector<uint8_t> typed_i64_dims;
    put_varint(typed_i64_dims, 1);
    std::vector<uint8_t> typed_i64;
    put_bytes_field(typed_i64, 1, typed_i64_dims);
    put_varint_field(typed_i64, 2, 7);
    put_packed_varint_field(typed_i64, 7, {6});
    put_string_field(typed_i64, 8, "typed_i64");

    std::vector<uint8_t> typed_f64_dims;
    put_varint(typed_f64_dims, 1);
    std::vector<uint8_t> typed_f64;
    put_bytes_field(typed_f64, 1, typed_f64_dims);
    put_varint_field(typed_f64, 2, 11);
    put_packed_double_field(typed_f64, 10, {0.75});
    put_string_field(typed_f64, 8, "typed_f64");

    const auto model = make_model_with_initializers(
        {raw_f32, raw_i32, raw_i64, raw_f64, typed_i32, typed_i64, typed_f64});

    auto parsed = parse_onnx_model(model);
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    ASSERT_EQ(parsed->graph.initializers.size(), 7u);
    EXPECT_EQ(parsed->graph.initializers[0].raw_data, encode_f32({1.5f, -2.5f}));
    EXPECT_TRUE(parsed->graph.initializers[0].double_data.empty());
    EXPECT_EQ(parsed->graph.initializers[3].raw_data, encode_f64({0.25}));
    EXPECT_TRUE(parsed->graph.initializers[3].double_data.empty());
    EXPECT_EQ(parsed->graph.initializers[5].int64_data,
              (std::vector<int64_t>{6}));
    EXPECT_EQ(parsed->graph.initializers[6].double_data,
              (std::vector<double>{0.75}));
    EXPECT_TRUE(parsed->graph.initializers[6].raw_data.empty());

    auto graph = load_wire_model(model, "raw_typed");
    ASSERT_TRUE(graph.has_value()) << graph.error().message;
    ASSERT_EQ(graph->weights.size(), 7u);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_f32").data<float>()[0], 1.5f);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_f32").data<float>()[1], -2.5f);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_i32").data<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_i32").data<float>()[1], -8.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_i64").data<float>()[0], 9.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("raw_f64").data<float>()[0], 0.25f);
    EXPECT_FLOAT_EQ(graph->weights.at("typed_i32").data<float>()[0], 4.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("typed_i32").data<float>()[1], 5.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("typed_i64").data<float>()[0], 6.0f);
    EXPECT_FLOAT_EQ(graph->weights.at("typed_f64").data<float>()[0], 0.75f);
}

TEST(OnnxLoaderTest, ParsesUnpackedAndPackedAttributeLists) {
    std::vector<uint8_t> pads_attr;
    put_string_field(pads_attr, 1, "pads");
    for (uint64_t pad : {1u, 1u, 1u, 1u}) {
        put_varint_field(pads_attr, 8, pad);
    }

    std::vector<uint8_t> strides_attr;
    put_string_field(strides_attr, 1, "strides");
    put_packed_varint_field(strides_attr, 8, {2, 2});

    std::vector<uint8_t> kernel_attr;
    put_string_field(kernel_attr, 1, "kernel_shape");
    for (uint64_t dim : {3u, 3u}) {
        put_varint_field(kernel_attr, 8, dim);
    }

    std::vector<uint8_t> coefficients_attr;
    put_string_field(coefficients_attr, 1, "coefficients");
    put_fixed32_field(coefficients_attr, 7, 0.5f);
    put_fixed32_field(coefficients_attr, 7, 2.5f);

    std::vector<uint8_t> scales_attr;
    put_string_field(scales_attr, 1, "scales");
    put_packed_float_field(scales_attr, 7, {4.5f, 5.5f});

    std::vector<uint8_t> node;
    put_string_field(node, 4, "Conv");
    put_bytes_field(node, 5, pads_attr);
    put_bytes_field(node, 5, strides_attr);
    put_bytes_field(node, 5, kernel_attr);
    put_bytes_field(node, 5, coefficients_attr);
    put_bytes_field(node, 5, scales_attr);

    std::vector<uint8_t> graph;
    put_bytes_field(graph, 1, node);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);

    auto parsed = parse_onnx_model(model);
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    ASSERT_EQ(parsed->graph.nodes.size(), 1u);
    const auto& attributes = parsed->graph.nodes[0].attributes;
    ASSERT_EQ(attributes.size(), 5u);

    const auto* pads = std::get_if<std::vector<int64_t>>(&attributes[0].value);
    ASSERT_NE(pads, nullptr);
    EXPECT_EQ(*pads, (std::vector<int64_t>{1, 1, 1, 1}));
    const auto* strides =
        std::get_if<std::vector<int64_t>>(&attributes[1].value);
    ASSERT_NE(strides, nullptr);
    EXPECT_EQ(*strides, (std::vector<int64_t>{2, 2}));
    const auto* kernel =
        std::get_if<std::vector<int64_t>>(&attributes[2].value);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(*kernel, (std::vector<int64_t>{3, 3}));
    const auto* coefficients =
        std::get_if<std::vector<double>>(&attributes[3].value);
    ASSERT_NE(coefficients, nullptr);
    EXPECT_EQ(*coefficients, (std::vector<double>{0.5, 2.5}));
    const auto* scales =
        std::get_if<std::vector<double>>(&attributes[4].value);
    ASSERT_NE(scales, nullptr);
    EXPECT_EQ(*scales, (std::vector<double>{4.5, 5.5}));
}

TEST(OnnxLoaderTest, MapsReduceModesAndSkipsOptionalNames) {
    std::vector<uint8_t> mean_node;
    put_string_field(mean_node, 1, "x");
    put_string_field(mean_node, 2, "y_mean");
    put_string_field(mean_node, 4, "ReduceMean");

    std::vector<uint8_t> sum_node;
    put_string_field(sum_node, 1, "x");
    put_string_field(sum_node, 2, "y_sum");
    put_string_field(sum_node, 4, "ReduceSum");

    std::vector<uint8_t> relay_node;
    put_string_field(relay_node, 1, "");
    put_string_field(relay_node, 1, "x");
    put_string_field(relay_node, 2, "");
    put_string_field(relay_node, 2, "y_relu");
    put_string_field(relay_node, 4, "Relu");

    std::vector<uint8_t> graph;
    put_bytes_field(graph, 1, mean_node);
    put_bytes_field(graph, 1, sum_node);
    put_bytes_field(graph, 1, relay_node);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);

    auto graph_result = load_wire_model(model, "reduce_optional");
    ASSERT_TRUE(graph_result.has_value()) << graph_result.error().message;
    ASSERT_EQ(graph_result->nodes.size(), 3u);

    EXPECT_EQ(graph_result->nodes[0].op, OpType::Reduce);
    const auto* mean_mode =
        std::get_if<std::string>(&graph_result->nodes[0].attrs.at("reduce_mode"));
    ASSERT_NE(mean_mode, nullptr);
    EXPECT_EQ(*mean_mode, "mean");

    EXPECT_EQ(graph_result->nodes[1].op, OpType::Reduce);
    const auto* sum_mode =
        std::get_if<std::string>(&graph_result->nodes[1].attrs.at("reduce_mode"));
    ASSERT_NE(sum_mode, nullptr);
    EXPECT_EQ(*sum_mode, "sum");

    EXPECT_EQ(graph_result->nodes[2].op, OpType::Relu);
    ASSERT_EQ(graph_result->nodes[2].inputs.size(), 1u);
    ASSERT_EQ(graph_result->nodes[2].outputs.size(), 1u);
    ASSERT_EQ(graph_result->edges.size(), 3u);
    EXPECT_EQ(graph_result->edges[2].value,
              graph_result->find_value("x")->id);
    EXPECT_EQ(graph_result->find_value("y_relu")->id,
              graph_result->nodes[2].outputs[0]);
}

TEST(OnnxLoaderTest, InitializerListedAsGraphInputIsNotInput) {
    std::vector<uint8_t> dims;
    put_varint(dims, 2);
    std::vector<uint8_t> tensor;
    put_bytes_field(tensor, 1, dims);
    put_varint_field(tensor, 2, 1);
    put_packed_float_field(tensor, 4, {1.0f, 2.0f});
    put_string_field(tensor, 8, "w");

    std::vector<uint8_t> input_info;
    put_string_field(input_info, 1, "w");

    std::vector<uint8_t> graph;
    put_bytes_field(graph, 5, tensor);
    put_bytes_field(graph, 11, input_info);
    std::vector<uint8_t> model;
    put_bytes_field(model, 7, graph);

    auto graph_result = load_wire_model(model, "initializer_input");
    ASSERT_TRUE(graph_result.has_value()) << graph_result.error().message;
    ASSERT_EQ(graph_result->weights.count("w"), 1u);
    ASSERT_NE(graph_result->find_value("w"), nullptr);
    EXPECT_FALSE(graph_result->find_value("w")->is_input);
    EXPECT_TRUE(graph_result->find_value("w")->shape == (Shape{{2}}));
}

TEST(OnnxMappingTest, MapsTraditionalWhitelistAndRejectsUnknown) {
    OnnxMapping mapping;
    EXPECT_EQ(*mapping.map("Conv"), OpType::Conv2D);
    EXPECT_EQ(*mapping.map("GlobalAveragePool"), OpType::GlobalAvgPool);
    EXPECT_EQ(*mapping.map("Flatten"), OpType::Flatten);
    EXPECT_EQ(mapping.map("DefinitelyNotAnOp").error().code,
              Error::Code::Unsupported);
}
