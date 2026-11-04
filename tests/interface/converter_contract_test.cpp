#include <gtest/gtest.h>

#include <memory>

#include "converter/format_loader/format_loader.h"
#include "converter/model_converter.h"
#include "converter/op_mapping/op_mapping.h"
#include "converter/quantization/quant_plugins.h"
#include "converter/validation/conversion_validator.h"
#include "core/plugin/plugin_registry.h"

using namespace ie;

TEST(ConverterContractsTest, DefaultsAreUnsupported) {
    EXPECT_EQ(ModelConverter::convert_onnx_to_ie("a", "b").error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(ConversionValidator::validate_structure(Graph{}, Graph{})
                  .error().code,
              Error::Code::Unsupported);

    OnnxLoader loader;
    EXPECT_EQ(loader.load("m", ModelLoaderConfig{}).error().code,
              Error::Code::Unsupported);
    OnnxMapping mapping;
    EXPECT_EQ(mapping.map("Conv").error().code, Error::Code::Unsupported);

    Int8QuantPlugin quant;
    Tensor tensor;
    EXPECT_EQ(quant.quantize(tensor, QuantConfig{}).error().code,
              Error::Code::Unsupported);
    EXPECT_EQ(quant.dequantize(tensor, QuantConfig{}).error().code,
              Error::Code::Unsupported);
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
