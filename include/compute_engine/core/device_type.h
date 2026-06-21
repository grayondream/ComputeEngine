#pragma once

#include "compute_engine/core/macros.h"
#include <cstdint>
#include <string>
#include <vector>

namespace compute_engine {

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA,
    OPENCL,
    VULKAN,
    METAL,
    NPU,
    DSP,
};

inline std::string DeviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::VULKAN: return "Vulkan";
        case DeviceType::METAL: return "Metal";
        case DeviceType::NPU: return "NPU";
        case DeviceType::DSP: return "DSP";
    }
    return "Unknown";
}

struct DeviceInfo {
    std::string name = "Unknown";
    std::string version = "0.0";
    size_t memory_total = 0;
    size_t memory_free = 0;
    DeviceType type = DeviceType::CPU;
    bool has_tensor_core = false;
};

} // namespace compute_engine
