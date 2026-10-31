#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace ie {

class CapabilitySet {
public:
    bool has(const std::string& name) const {
        auto it = caps_.find(name);
        return it != caps_.end() && it->second;
    }

    void set(const std::string& name, bool value) { caps_[name] = value; }

    std::vector<std::string> list() const {
        std::vector<std::string> names;
        names.reserve(caps_.size());
        for (const auto& [name, enabled] : caps_) {
            if (enabled) {
                names.push_back(name);
            }
        }
        std::sort(names.begin(), names.end());
        return names;
    }

private:
    std::unordered_map<std::string, bool> caps_;
};

}  // namespace ie
