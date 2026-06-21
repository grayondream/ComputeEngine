#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace compute_engine {

struct float16 {
    uint16_t bits;

    float16() : bits(0) {}
    explicit float16(float f);
    explicit operator float() const;
};

struct bfloat16 {
    uint16_t bits;

    bfloat16() : bits(0) {}
    explicit bfloat16(float f);
    explicit operator float() const;
};

inline float16 FloatToHalf(float f) {
    float16 h;
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = (x >> 13) & 0x3ff;
    if (exp <= 0) { h.bits = 0; return h; }
    if (exp >= 31) { h.bits = sign | 0x7c00; return h; }
    h.bits = static_cast<uint16_t>(sign | (exp << 10) | mantissa);
    return h;
}

inline float HalfToFloat(float16 h) {
    uint32_t sign = (h.bits & 0x8000) << 16;
    int32_t exp = ((h.bits >> 10) & 0x1f) - 15 + 127;
    uint32_t mantissa = (h.bits & 0x3ff) << 13;
    if (exp <= 0) {
        uint32_t val = sign;
        std::memcpy(&val, &val, 4);
        return *reinterpret_cast<float*>(&val);
    }
    if (exp >= 255) { exp = 255; mantissa = 0; }
    uint32_t x = sign | (static_cast<uint32_t>(exp) << 23) | mantissa;
    float result;
    std::memcpy(&result, &x, 4);
    return result;
}

inline float16::float16(float f) : bits(FloatToHalf(f).bits) {}
inline float16::operator float() const { float16 h; h.bits = bits; return HalfToFloat(h); }

inline bfloat16 FloatToBFloat16(float f) {
    bfloat16 b;
    uint32_t x;
    std::memcpy(&x, &f, 4);
    b.bits = static_cast<uint16_t>(x >> 16);
    return b;
}

inline float BFloat16ToFloat(bfloat16 b) {
    uint32_t x = static_cast<uint32_t>(b.bits) << 16;
    float result;
    std::memcpy(&result, &x, 4);
    return result;
}

inline bfloat16::bfloat16(float f) : bits(FloatToBFloat16(f).bits) {}
inline bfloat16::operator float() const { bfloat16 b; b.bits = bits; return BFloat16ToFloat(b); }

} // namespace compute_engine
