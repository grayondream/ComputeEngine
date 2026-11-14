#include "converter/model_converter.h"

#include <string>

#include "models/traditional/onnx/onnx_loader.h"

namespace ie {

Result<Graph> ModelConverter::load_model(const std::string& path,
                                         const ModelLoaderConfig& config) {
    switch (config.format) {
        case ModelFormat::ONNX:
            return load_onnx_model(path);
        default:
            return Error::unsupported(
                "ModelConverter::load_model supports ONNX models only");
    }
}

}  // namespace ie
