#include "core/base/types.h"

namespace ie {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return 4;
        case DType::F16:
        case DType::BF16:
            return 2;
        case DType::I8:
        case DType::U8:
        case DType::FP8_E4M3:
        case DType::FP8_E5M2:
        case DType::BOOL:
            return 1;
        case DType::I4:
        case DType::MXFP4:
            return 1;  // packed nibbles are addressed per byte at this layer
        case DType::MXFP6:
            return 1;
    }
    return 0;
}

std::string to_string(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return "F32";
        case DType::F16:
            return "F16";
        case DType::BF16:
            return "BF16";
        case DType::I8:
            return "I8";
        case DType::I4:
            return "I4";
        case DType::U8:
            return "U8";
        case DType::FP8_E4M3:
            return "FP8_E4M3";
        case DType::FP8_E5M2:
            return "FP8_E5M2";
        case DType::MXFP4:
            return "MXFP4";
        case DType::MXFP6:
            return "MXFP6";
        case DType::BOOL:
            return "BOOL";
    }
    return "UNKNOWN";
}

std::string to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
        case Device::VULKAN:
            return "VULKAN";
    }
    return "UNKNOWN";
}

}  // namespace ie
