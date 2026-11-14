#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/base/error.h"
#include "core/ir/attr.h"

namespace ie {
namespace detail {

// ONNX auto_pad values. "NOTSET" and a missing attribute both mean that the
// explicit pads attribute is authoritative.
enum class AutoPad { NotSet, Valid, SameUpper, SameLower };

struct IntListAttr {
    bool found{false};
    std::vector<int64_t> values;
};

struct IntAttr {
    bool found{false};
    int64_t value{0};
};

struct DoubleAttr {
    bool found{false};
    double value{0.0};
};

struct StringAttr {
    bool found{false};
    std::string value;
};

inline Status read_int_list(const Attrs& attrs, const char* key,
                            IntListAttr& out) {
    out = IntListAttr{};
    const auto it = attrs.find(key);
    if (it == attrs.end()) {
        return {};
    }
    const auto* values = std::get_if<std::vector<int64_t>>(&it->second);
    if (values == nullptr) {
        return Error::invalid_argument(std::string("attribute '") + key +
                                       "' must be an int list");
    }
    out.found = true;
    out.values = *values;
    return {};
}

inline Status read_int(const Attrs& attrs, const char* key, IntAttr& out) {
    out = IntAttr{};
    const auto it = attrs.find(key);
    if (it == attrs.end()) {
        return {};
    }
    if (const auto* value = std::get_if<int64_t>(&it->second)) {
        out = {true, *value};
        return {};
    }
    if (const auto* value = std::get_if<bool>(&it->second)) {
        out = {true, *value ? 1 : 0};
        return {};
    }
    return Error::invalid_argument(std::string("attribute '") + key +
                                   "' must be an int");
}

inline Status read_double(const Attrs& attrs, const char* key, DoubleAttr& out) {
    out = DoubleAttr{};
    const auto it = attrs.find(key);
    if (it == attrs.end()) {
        return {};
    }
    if (const auto* value = std::get_if<double>(&it->second)) {
        out = {true, *value};
        return {};
    }
    if (const auto* value = std::get_if<int64_t>(&it->second)) {
        out = {true, static_cast<double>(*value)};
        return {};
    }
    return Error::invalid_argument(std::string("attribute '") + key +
                                   "' must be numeric");
}

inline Status read_string(const Attrs& attrs, const char* key,
                          StringAttr& out) {
    out = StringAttr{};
    const auto it = attrs.find(key);
    if (it == attrs.end()) {
        return {};
    }
    const auto* value = std::get_if<std::string>(&it->second);
    if (value == nullptr) {
        return Error::invalid_argument(std::string("attribute '") + key +
                                       "' must be a string");
    }
    out = {true, *value};
    return {};
}

inline Status read_auto_pad(const Attrs& attrs, AutoPad& out) {
    out = AutoPad::NotSet;
    StringAttr value;
    if (auto status = read_string(attrs, "auto_pad", value);
        !status.has_value()) {
        return status;
    }
    if (!value.found || value.value.empty() || value.value == "NOTSET") {
        return {};
    }
    if (value.value == "VALID") {
        out = AutoPad::Valid;
    } else if (value.value == "SAME_UPPER") {
        out = AutoPad::SameUpper;
    } else if (value.value == "SAME_LOWER") {
        out = AutoPad::SameLower;
    } else {
        return Error::invalid_argument("unsupported auto_pad: " + value.value);
    }
    return {};
}

// Resolves the optional pads attribute to {top, left, bottom, right}. A
// two-element list is treated as symmetric padding; an empty/missing list is
// zero padding.
inline Status read_pads(const Attrs& attrs, std::array<int64_t, 4>& pads) {
    pads = {0, 0, 0, 0};
    IntListAttr value;
    if (auto status = read_int_list(attrs, "pads", value); !status.has_value()) {
        return status;
    }
    if (!value.found || value.values.empty()) {
        return {};
    }
    if (value.values.size() == 2) {
        pads = {value.values[0], value.values[1], value.values[0],
                value.values[1]};
    } else if (value.values.size() == 4) {
        pads = {value.values[0], value.values[1], value.values[2],
                value.values[3]};
    } else {
        return Error::invalid_argument("pads must contain 2 or 4 values");
    }
    for (int64_t pad : pads) {
        if (pad < 0) {
            return Error::invalid_argument("pads must be non-negative");
        }
    }
    return {};
}

// Reads a two-element positive int list (strides/dilations/kernel_shape).
inline Status read_pair(const Attrs& attrs, const char* key,
                        std::array<int64_t, 2>& out) {
    IntListAttr value;
    if (auto status = read_int_list(attrs, key, value); !status.has_value()) {
        return status;
    }
    if (!value.found) {
        return {};
    }
    if (value.values.size() != 2) {
        return Error::invalid_argument(std::string(key) +
                                       " must contain 2 values");
    }
    if (value.values[0] <= 0 || value.values[1] <= 0) {
        return Error::invalid_argument(std::string(key) +
                                       " values must be positive");
    }
    out = {value.values[0], value.values[1]};
    return {};
}

// SAME padding for one spatial axis. The output extent is ceil(input/stride);
// the leftover padding goes after the window for SAME_UPPER and before it for
// SAME_LOWER.
inline std::pair<int64_t, int64_t> same_padding(int64_t input, int64_t kernel,
                                                int64_t dilation,
                                                int64_t stride, bool lower) {
    const int64_t output = (input + stride - 1) / stride;
    const int64_t total = std::max<int64_t>(
        0, (output - 1) * stride + dilation * (kernel - 1) + 1 - input);
    int64_t before = total / 2;
    int64_t after = total - before;
    if (lower) {
        std::swap(before, after);
    }
    return {before, after};
}

inline int64_t ceil_div(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

}  // namespace detail
}  // namespace ie
