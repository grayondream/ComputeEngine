#pragma once

#include <memory>
#include <string>

#include "core/backend/backend.h"
#include "core/base/error.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

struct BackendConfig {
    std::string kind;
    std::string params;
};

// Plugin surface for loading backend implementations (docs/readme.md §12.1).
class BackendPlugin {
public:
    virtual ~BackendPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual Result<std::unique_ptr<Backend>> create(const BackendConfig& config) = 0;
};

}  // namespace ie
