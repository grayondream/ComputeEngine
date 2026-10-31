#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/backend/backend.h"

namespace ie {

class BackendRegistry {
public:
    using Factory = std::function<std::unique_ptr<Backend>()>;

    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }

    Status register_backend(BackendKind kind, Factory factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factories_.find(kind) != factories_.end()) {
            return Error::already_exists("backend already registered");
        }
        factories_.emplace(kind, std::move(factory));
        return {};
    }

    Result<std::unique_ptr<Backend>> create(BackendKind kind) {
        Factory factory;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = factories_.find(kind);
            if (it == factories_.end()) {
                return Error::not_found("backend not registered");
            }
            factory = it->second;
        }
        return factory();
    }

    bool contains(BackendKind kind) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factories_.find(kind) != factories_.end();
    }

    std::vector<BackendKind> list() const {
        static constexpr BackendKind kOrder[] = {BackendKind::CPU,
                                                 BackendKind::CUDA,
                                                 BackendKind::VULKAN};
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<BackendKind> kinds;
        for (BackendKind kind : kOrder) {
            if (factories_.find(kind) != factories_.end()) {
                kinds.push_back(kind);
            }
        }
        return kinds;
    }

private:
    BackendRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<BackendKind, Factory> factories_;
};

}  // namespace ie
