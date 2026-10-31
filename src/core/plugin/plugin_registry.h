#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/base/error.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

template <class PluginT>
class PluginRegistry {
public:
    static PluginRegistry& instance() {
        static PluginRegistry registry;
        return registry;
    }

    Status register_plugin(std::unique_ptr<PluginT> plugin) {
        if (!plugin) {
            return Error::invalid_argument("null plugin");
        }
        const std::string name = plugin->meta().name;
        if (plugins_.find(name) != plugins_.end()) {
            return Error::already_exists("plugin already registered: " + name);
        }
        plugins_.emplace(name, std::move(plugin));
        return {};
    }

    Result<PluginT*> get(const std::string& name) {
        auto it = plugins_.find(name);
        if (it == plugins_.end()) {
            return Error::not_found("plugin not found: " + name);
        }
        return it->second.get();
    }

    std::vector<PluginMeta> list() const {
        std::vector<PluginMeta> metas;
        metas.reserve(plugins_.size());
        for (const auto& [name, plugin] : plugins_) {
            (void)name;
            metas.push_back(plugin->meta());
        }
        return metas;
    }

    std::vector<PluginMeta> list_by_tag(const std::string& tag) const {
        std::vector<PluginMeta> metas;
        for (const auto& [name, plugin] : plugins_) {
            (void)name;
            PluginMeta meta = plugin->meta();
            for (const std::string& candidate : meta.tags) {
                if (candidate == tag) {
                    metas.push_back(std::move(meta));
                    break;
                }
            }
        }
        return metas;
    }

    Status load_shared_library(const std::string& /*path*/) {
        return Error::unsupported("plugin loading not implemented in P0");
    }

private:
    PluginRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<PluginT>> plugins_;
};

}  // namespace ie
