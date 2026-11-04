#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/base/error.h"
#include "core/ir/pass.h"

namespace ie {

// Global factory registry for named passes. Registration rejects empty names,
// empty creators, and duplicate names.
class PassRegistry {
public:
    using Creator = std::function<std::unique_ptr<Pass>()>;

    static PassRegistry& instance() {
        static PassRegistry registry;
        return registry;
    }

    Status register_pass(std::string name, Creator creator) {
        if (name.empty()) {
            return Error::invalid_argument("pass name must not be empty");
        }
        if (!creator) {
            return Error::invalid_argument(
                "pass creator must not be empty: " + name);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (creators_.find(name) != creators_.end()) {
            return Error::already_exists("pass already registered: " + name);
        }
        creators_.emplace(std::move(name), std::move(creator));
        return {};
    }

    Result<std::unique_ptr<Pass>> create(std::string_view name) {
        Creator creator;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = creators_.find(std::string(name));
            if (it == creators_.end()) {
                return Error::not_found("pass not found: " + std::string(name));
            }
            creator = it->second;
        }
        std::unique_ptr<Pass> pass = creator();
        if (!pass) {
            return Error::runtime(
                "pass factory returned null: " + std::string(name));
        }
        return Result<std::unique_ptr<Pass>>(std::move(pass));
    }

    std::vector<std::string> list() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(creators_.size());
        for (const auto& [name, creator] : creators_) {
            (void)creator;
            names.push_back(name);
        }
        return names;
    }

private:
    PassRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Creator> creators_;
};

}  // namespace ie
