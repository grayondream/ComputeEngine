#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "converter/format_loader/format_loader.h"
#include "converter/model_converter.h"
#include "converter/op_mapping/op_mapping.h"
#include "converter/quantization/quant_plugins.h"
#include "converter/validation/conversion_validator.h"
#include "core/plugin/plugin_registry.h"

using namespace ie;

namespace {

using UnsupportedCheck = std::function<Error::Code()>;

void expect_all_unsupported(
    const std::vector<std::pair<const char*, UnsupportedCheck>>& cases) {
    for (const auto& [name, check] : cases) {
        SCOPED_TRACE(name);
        EXPECT_EQ(check(), Error::Code::Unsupported);
    }
}

}  // namespace

TEST(ConverterContractsTest, DefaultsAreUnsupported) {
    EXPECT_EQ(ModelConverter::convert_onnx_to_ie("a", "b").error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(ConversionValidator::validate_structure(Graph{}, Graph{})
                  .error().code,
              Error::Code::Unsupported);

    Int8QuantPlugin quant;
    Tensor tensor;
    EXPECT_EQ(quant.quantize(tensor, QuantConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(quant.dequantize(tensor, QuantConfig{}).error().code,
              Error::Code::Unsupported);
}

TEST(ConverterContractsTest, AllModelConverterMethodsUnsupported) {
    expect_all_unsupported({
        {"convert_onnx_to_ie",
         [] { return ModelConverter::convert_onnx_to_ie("m.onnx", "m.ie").error().code; }},
        {"convert_tflite_to_ie",
         [] { return ModelConverter::convert_tflite_to_ie("m.tflite", "m.ie").error().code; }},
        {"convert_hf_to_safetensors",
         [] { return ModelConverter::convert_hf_to_safetensors("hf", "m.safetensors").error().code; }},
        {"convert_hf_to_gguf",
         [] { return ModelConverter::convert_hf_to_gguf("hf", "m.gguf").error().code; }},
        {"load_model",
         [] { return ModelConverter::load_model("m", ModelLoaderConfig{}).error().code; }},
    });
}

TEST(ConverterContractsTest, AllFormatLoadersUnsupported) {
    expect_all_unsupported({
        {"TfliteLoader",
         [] { return TfliteLoader{}.load("m", ModelLoaderConfig{}).error().code; }},
        {"CaffeLoader",
         [] { return CaffeLoader{}.load("m", ModelLoaderConfig{}).error().code; }},
        {"SafetensorsLoader",
         [] { return SafetensorsLoader{}.load("m", ModelLoaderConfig{}).error().code; }},
        {"GgufLoader",
         [] { return GgufLoader{}.load("m", ModelLoaderConfig{}).error().code; }},
    });
}

TEST(ConverterContractsTest, AllOpMappingsUnsupported) {
    expect_all_unsupported({
        {"TfliteMapping", [] { return TfliteMapping{}.map("Conv").error().code; }},
        {"PyTorchMapping",
         [] { return PyTorchMapping{}.map("Conv").error().code; }},
        {"GgufMapping", [] { return GgufMapping{}.map("Conv").error().code; }},
    });
}

TEST(ConverterContractsTest, OnnxLoaderAndMappingAreImplemented) {
    auto graph = OnnxLoader{}.load(
        std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx", ModelLoaderConfig{});
    ASSERT_TRUE(graph.has_value()) << graph.error().message;
    ASSERT_EQ(graph->nodes.size(), 1u);
    EXPECT_EQ(graph->nodes[0].op, OpType::Add);
    ASSERT_EQ(graph->weights.count("w"), 1u);

    OnnxMapping mapping;
    EXPECT_EQ(*mapping.map("Conv"), OpType::Conv2D);
    EXPECT_EQ(*mapping.map("Flatten"), OpType::Flatten);
    EXPECT_EQ(mapping.map("DefinitelyNotAnOp").error().code,
              Error::Code::Unsupported);
}

TEST(ConverterContractsTest, AllConversionValidatorStagesUnsupported) {
    expect_all_unsupported({
        {"validate_structure",
         [] { return ConversionValidator::validate_structure(Graph{}, Graph{}).error().code; }},
        {"validate_weights",
         [] { return ConversionValidator::validate_weights("a", "b").error().code; }},
        {"validate_layer_outputs",
         [] {
             return ConversionValidator::validate_layer_outputs("a", "b", {}).error().code;
         }},
        {"validate_end_to_end",
         [] {
             return ConversionValidator::validate_end_to_end("a", "b", {}).error().code;
         }},
        {"validate_tokenizer_metadata",
         [] {
             return ConversionValidator::validate_tokenizer_metadata("a", "b").error().code;
         }},
    });
}

TEST(QuantPluginMetaTest, ApiVersionIsFrozenAtZeroOne) {
    EXPECT_EQ(Int8QuantPlugin{}.meta().api_version, "0.1");
    EXPECT_EQ(GgufQuantPlugin{}.meta().api_version, "0.1");
    EXPECT_EQ(AwqQuantPlugin{}.meta().api_version, "0.1");
    EXPECT_EQ(GptqQuantPlugin{}.meta().api_version, "0.1");
}

TEST(QuantPluginRegistryTest, CompatiblePluginRegisters) {
    auto& registry = PluginRegistry<QuantPlugin>::instance();
    auto status = registry.register_plugin(std::make_unique<Int8QuantPlugin>());
    EXPECT_TRUE(status.has_value() || status.error().code == Error::Code::AlreadyExists);
}
