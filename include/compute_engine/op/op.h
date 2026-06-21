#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/dtype.h"
#include <cstdint>
#include <string>

namespace compute_engine {

enum class OpType : uint16_t {
    // Arithmetic
    ADD = 0, SUB, MUL, DIV, NEG, POW, SQRT,
    // Linear algebra
    MATMUL, BATCH_MATMUL, TRANSPOSE, RESHAPE,
    // Convolution & Pooling
    CONV1D, CONV2D, CONV3D, CONV_TRANSPOSE,
    MAX_POOL, AVG_POOL, GLOBAL_POOL,
    // Normalization
    BATCH_NORM, LAYER_NORM, RMS_NORM, INSTANCE_NORM, GROUP_NORM,
    // Activation
    RELU, LEAKY_RELU, GELU, FAST_GELU, SILU, SIGMOID, TANH, SOFTMAX,
    // LLM specific
    ATTENTION, ROPE, KV_CACHE_SAVE, KV_CACHE_LOAD, SWIGLU, MLA,
    // Other
    CONCAT, SLICE, CAST, PAD, GATHER, SCATTER,
    // Custom
    CUSTOM = 1000,
};

inline std::string OpTypeToString(OpType op) {
    static const char* names[] = {
        "Add", "Sub", "Mul", "Div", "Neg", "Pow", "Sqrt",
        "MatMul", "BatchMatMul", "Transpose", "Reshape",
        "Conv1D", "Conv2D", "Conv3D", "ConvTranspose",
        "MaxPool", "AvgPool", "GlobalPool",
        "BatchNorm", "LayerNorm", "RMSNorm", "InstanceNorm", "GroupNorm",
        "ReLU", "LeakyReLU", "GELU", "FastGELU", "SiLU", "Sigmoid", "Tanh", "Softmax",
        "Attention", "RoPE", "KVCacheSave", "KVCacheLoad", "SwiGLU", "MLA",
        "Concat", "Slice", "Cast", "Pad", "Gather", "Scatter",
    };
    auto idx = static_cast<uint16_t>(op);
    if (idx < sizeof(names) / sizeof(names[0])) return names[idx];
    return "Custom_" + std::to_string(idx);
}

} // namespace compute_engine
