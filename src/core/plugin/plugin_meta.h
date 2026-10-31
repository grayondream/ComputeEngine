#pragma once

#include <string>
#include <vector>

namespace ie {

struct PluginMeta {
    std::string name;
    std::string version;
    std::string api_version;
    std::vector<std::string> tags;
    std::string description;
};

}  // namespace ie
