#pragma once

#include "core/plugin/quant_plugin.h"

namespace ie {

class Int8QuantPlugin final : public QuantPlugin {
public:
    PluginMeta meta() const override {
        return {"int8_quant", "0.1.0", "0.1", {"quantization"}, ""};
    }

    Result<Tensor> quantize(const Tensor& /*tensor*/,
                            const QuantConfig& /*config*/) override {
        return Error::unsupported("int8 quantization not implemented in P0.5");
    }

    Result<Tensor> dequantize(const Tensor& /*tensor*/,
                              const QuantConfig& /*config*/) override {
        return Error::unsupported("int8 dequantization not implemented in P0.5");
    }
};

class GgufQuantPlugin final : public QuantPlugin {
public:
    PluginMeta meta() const override {
        return {"gguf_quant", "0.1.0", "0.1", {"quantization"}, ""};
    }

    Result<Tensor> quantize(const Tensor& /*tensor*/,
                            const QuantConfig& /*config*/) override {
        return Error::unsupported("gguf quantization not implemented in P0.5");
    }

    Result<Tensor> dequantize(const Tensor& /*tensor*/,
                              const QuantConfig& /*config*/) override {
        return Error::unsupported("gguf dequantization not implemented in P0.5");
    }
};

class AwqQuantPlugin final : public QuantPlugin {
public:
    PluginMeta meta() const override {
        return {"awq_quant", "0.1.0", "0.1", {"quantization"}, ""};
    }

    Result<Tensor> quantize(const Tensor& /*tensor*/,
                            const QuantConfig& /*config*/) override {
        return Error::unsupported("awq quantization not implemented in P0.5");
    }

    Result<Tensor> dequantize(const Tensor& /*tensor*/,
                              const QuantConfig& /*config*/) override {
        return Error::unsupported("awq dequantization not implemented in P0.5");
    }
};

class GptqQuantPlugin final : public QuantPlugin {
public:
    PluginMeta meta() const override {
        return {"gptq_quant", "0.1.0", "0.1", {"quantization"}, ""};
    }

    Result<Tensor> quantize(const Tensor& /*tensor*/,
                            const QuantConfig& /*config*/) override {
        return Error::unsupported("gptq quantization not implemented in P0.5");
    }

    Result<Tensor> dequantize(const Tensor& /*tensor*/,
                              const QuantConfig& /*config*/) override {
        return Error::unsupported("gptq dequantization not implemented in P0.5");
    }
};

}  // namespace ie
