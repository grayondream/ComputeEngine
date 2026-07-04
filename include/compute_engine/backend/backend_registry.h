#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/device_type.h"
#include <vector>
#include <functional>
#include <mutex>
#include <memory>

namespace compute_engine {

class BackendRegistry {
public:
    static BackendRegistry& Instance();

    void Register(DeviceType type,
                  std::function<BackendPtr()> factory,
                  int priority = 100);

    BackendPtr Get(DeviceType type);

    BackendPtr GetBestAvailable(const std::vector<DeviceType>& preferences);

    BackendPtr GetBestAvailable();

    std::vector<BackendPtr> GetAllAvailable();

    void SetPriorityList(const std::vector<DeviceType>& ordered);

    static void InitBuiltinBackends();

private:
    BackendRegistry() = default;

    struct Entry {
        DeviceType type;
        std::function<BackendPtr()> factory;
        int priority;
    };
    std::vector<Entry> entries_;
    std::vector<DeviceType> priority_list_;
    std::mutex mutex_;
};

#define REGISTER_BACKEND(type, factory, priority)               \
    static struct AutoRegister_##type {                         \
        AutoRegister_##type() {                                 \
            BackendRegistry::Instance().Register(               \
                type, factory, priority);                       \
        }                                                       \
    } s_auto_reg_##type

} // namespace compute_engine
