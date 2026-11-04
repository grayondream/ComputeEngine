#pragma once

#include <string>

#include "core/base/error.h"
#include "core/plugin/plugin_api.h"

namespace ie {

// SDK metadata surfaced to bindings and CLIs (docs/readme.md §20.18).
struct SdkInfo {
    std::string name{"ComputeEngine"};
    std::string version{"0.1.0"};
    std::string api_version{kApiVersionString};
    std::string build_type;
};

inline SdkInfo sdk_info() { return {}; }

class Sdk {
public:
    static Status initialize() { return {}; }
    static Status shutdown() { return {}; }
    static SdkInfo info() { return sdk_info(); }
};

}  // namespace ie
