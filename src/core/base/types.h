#pragma once

#include <cstddef>
#include <string>

namespace ie {

enum class Device { CPU, CUDA, VULKAN };

enum class DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    U8,
    FP8_E4M3,
    FP8_E5M2,
    MXFP4,
    MXFP6,
    BOOL,
};

enum class Layout { ROW_MAJOR, COL_MAJOR, BLOCKED, NHWC, NCHW };

std::size_t dtype_size(DType dtype);
std::string to_string(DType dtype);
std::string to_string(Device device);

}  // namespace ie
