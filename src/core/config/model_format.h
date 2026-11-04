#pragma once

namespace ie {

// Model container formats supported by the converter / model loader.
enum class ModelFormat {
    ONNX,
    TFLITE,
    CAFFE,
    SAFETENSORS,
    GGUF,
    CUSTOM,
};

}  // namespace ie
