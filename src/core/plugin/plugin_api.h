#pragma once

#include <charconv>
#include <string_view>
#include <system_error>

namespace ie {

// Frozen plugin API version (docs/readme.md §12.4). A plugin is compatible
// when its major version matches and its minor version is not newer.
inline constexpr int kApiVersionMajor = 0;
inline constexpr int kApiVersionMinor = 1;
inline constexpr std::string_view kApiVersion = "0.1";

namespace detail {

inline bool parse_version_number(std::string_view text, int& value) {
    if (text.empty()) {
        return false;
    }
    for (const char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

}  // namespace detail

// Accepts exactly "<digits>.<digits>"; anything else (including empty,
// negative, or extra characters) is incompatible.
inline bool is_api_compatible(std::string_view version) {
    const std::size_t dot = version.find('.');
    if (dot == std::string_view::npos) {
        return false;
    }
    const std::string_view major_text = version.substr(0, dot);
    const std::string_view minor_text = version.substr(dot + 1);
    if (minor_text.find('.') != std::string_view::npos) {
        return false;
    }

    int major = 0;
    int minor = 0;
    if (!detail::parse_version_number(major_text, major) ||
        !detail::parse_version_number(minor_text, minor)) {
        return false;
    }
    return major == kApiVersionMajor && minor <= kApiVersionMinor;
}

}  // namespace ie
