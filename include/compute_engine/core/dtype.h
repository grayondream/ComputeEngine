#pragma once

#include "compute_engine/core/macros.h"
#include <cstdint>
#include <string>

namespace compute_engine {

enum class DType : uint8_t {
    FLOAT32 = 0,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT8,
    INT4,
    UINT8,
    BOOL,
};

inline size_t SizeOf(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT16: return 2;
        case DType::BFLOAT16: return 2;
        case DType::INT32: return 4;
        case DType::INT8: return 1;
        case DType::INT4: return 1; // packed
        case DType::UINT8: return 1;
        case DType::BOOL: return 1;
    }
    return 0;
}

inline bool IsFloat(DType dtype) {
    return dtype == DType::FLOAT32 || dtype == DType::FLOAT16 || dtype == DType::BFLOAT16;
}

inline bool IsInteger(DType dtype) {
    return dtype == DType::INT32 || dtype == DType::INT8 || dtype == DType::INT4 || dtype == DType::UINT8;
}

inline std::string DTypeToString(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return "float32";
        case DType::FLOAT16: return "float16";
        case DType::BFLOAT16: return "bfloat16";
        case DType::INT32: return "int32";
        case DType::INT8: return "int8";
        case DType::INT4: return "int4";
        case DType::UINT8: return "uint8";
        case DType::BOOL: return "bool";
    }
    return "unknown";
}

} // namespace compute_engine
