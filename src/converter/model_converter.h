#pragma once

#include <string>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "llm/model_loader/model_loader.h"

namespace ie {

// GGUF quantization schemes exposed by the HF -> GGUF converter
// (docs/readme.md §11.2).
enum class GGUFQuantType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q4_K_M,
    Q4_0,
    Q3_K_M,
    Q2_K
};

// Model conversion entry points (docs/readme.md §11.2). P0.5 freezes the
// contract only; every conversion path is unsupported.
class ModelConverter {
public:
    static Status convert_onnx_to_ie(const std::string& /*onnx_path*/,
                                     const std::string& /*output_path*/) {
        return Error::unsupported("onnx conversion not implemented in P0.5");
    }

    static Status convert_tflite_to_ie(const std::string& /*tflite_path*/,
                                       const std::string& /*output_path*/) {
        return Error::unsupported("tflite conversion not implemented in P0.5");
    }

    static Status convert_hf_to_safetensors(
        const std::string& /*hf_path*/, const std::string& /*output_path*/) {
        return Error::unsupported(
            "safetensors conversion not implemented in P0.5");
    }

    static Status convert_hf_to_gguf(
        const std::string& /*hf_path*/, const std::string& /*output_path*/,
        GGUFQuantType /*quant*/ = GGUFQuantType::F16) {
        return Error::unsupported("gguf conversion not implemented in P0.5");
    }

    static Result<Graph> load_model(const std::string& /*path*/,
                                    const ModelLoaderConfig& /*config*/) {
        return Error::unsupported("model loading not implemented in P0.5");
    }
};

}  // namespace ie
