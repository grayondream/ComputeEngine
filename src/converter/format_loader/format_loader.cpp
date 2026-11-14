#include "converter/format_loader/format_loader.h"

#include <string>

#include "models/traditional/onnx/onnx_loader.h"

namespace ie {

Result<Graph> OnnxLoader::load(const std::string& path,
                               const ModelLoaderConfig& /*config*/) {
    return load_onnx_model(path);
}

}  // namespace ie
