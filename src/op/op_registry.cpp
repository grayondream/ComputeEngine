#include "compute_engine/op/op_registry.h"
#include <algorithm>
#include "compute_engine/platform/logger.h"

namespace compute_engine {

OpSchemaRegistry& OpSchemaRegistry::Instance() {
    static OpSchemaRegistry instance;
    LOGI("OpSchemaRegistry initialized");
    return instance;
}

void OpSchemaRegistry::Register(const std::string& op_type, OpSchema schema) {
    schema.name = op_type;
    schemas_[op_type] = std::move(schema);
    LOGI("Op registered: {}", op_type);
}

const OpSchema* OpSchemaRegistry::Get(const std::string& op_type) const {
    auto it = schemas_.find(op_type);
    if (it != schemas_.end()) {
        LOGI("Op schema lookup: {} -> found", op_type);
        return &it->second;
    }
    LOGW("Op schema lookup: {} -> not found", op_type);
    return nullptr;
}

bool OpSchemaRegistry::Has(const std::string& op_type) const {
    return schemas_.count(op_type) > 0;
}

std::vector<std::string> OpSchemaRegistry::ListOps() const {
    std::vector<std::string> result;
    for (auto& [k, v] : schemas_) result.push_back(k);
    return result;
}

KernelRegistry& KernelRegistry::Instance() {
    static KernelRegistry instance;
    LOGI("KernelRegistry initialized");
    return instance;
}

void KernelRegistry::Register(
    const std::string& op_type, DeviceType device,
    int priority, KernelFactory factory) {
    for (auto& [name, entries] : entries_) {
        if (name == op_type) {
            entries.push_back({device, priority, std::move(factory)});
            return;
        }
    }
    entries_.emplace_back(op_type, std::vector<Entry>{{device, priority, std::move(factory)}});
    LOGI("Kernel registered: op={} device={} priority={}", op_type, DeviceTypeToString(device), priority);
}

Status KernelRegistry::GetOrCompile(
    const std::string& op_type, KernelExecutor* exec,
    const KernelCacheKey& key, KernelHandle* handle) {
    for (auto& [k, h] : cache_) {
        if (k == key) {
            *handle = h;
            LOGI("Kernel cache hit for {} on {}", op_type, DeviceTypeToString(key.device));
            return Status::OK();
        }
    }
    for (auto& [name, entries] : entries_) {
        if (name == op_type) {
            for (auto& e : entries) {
                if (e.device == key.device) {
                    KernelHandle h;
                    auto status = e.factory(exec, &h);
                    if (status.ok()) {
                        cache_.emplace_back(key, h);
                        *handle = h;
                        LOGI("Kernel compiled for {} on {}", op_type, DeviceTypeToString(key.device));
                        return Status::OK();
                    }
                }
            }
        }
    }
    LOGW("Kernel not found for {} on {}", op_type, DeviceTypeToString(key.device));
    return Status::NotFound("No kernel for " + op_type + " on " + DeviceTypeToString(key.device));
}

bool KernelRegistry::Has(const std::string& op_type, DeviceType device) const {
    for (auto& [name, entries] : entries_) {
        if (name == op_type) {
            for (auto& e : entries) {
                if (e.device == device) return true;
            }
        }
    }
    return false;
}

void KernelRegistry::ClearCache() {
    LOGI("Kernel cache cleared ({} entries)", cache_.size());
    cache_.clear();
}

} // namespace compute_engine
