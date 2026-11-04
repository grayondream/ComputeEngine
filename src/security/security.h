#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/base/error.h"

namespace ie {

// Multi-tenant isolation (docs/readme.md §20.18). The surface is frozen in
// P0.5; enforcement arrives with the deployment milestone.
struct TenantConfig {
    std::string id;
    std::size_t max_memory_bytes{0};
    int max_concurrent_requests{0};
};

class TenantManager {
public:
    Status create_tenant(const TenantConfig& config) {
        (void)config;
        return Error::unsupported(
            "multi-tenant isolation not implemented in P0.5");
    }

    Status remove_tenant(std::string_view tenant_id) {
        (void)tenant_id;
        return Error::unsupported(
            "multi-tenant isolation not implemented in P0.5");
    }

    bool exists(std::string_view tenant_id) const {
        (void)tenant_id;
        return false;
    }

    std::vector<std::string> list() const { return {}; }
};

struct SandboxConfig {
    bool allow_network{false};
    bool allow_file_system{false};
    std::string root_dir;
};

// Sandboxing is not implemented in P0.5: no lifecycle method performs any
// enforcement, so all of them report Unsupported instead of pretending to
// succeed.
class Sandbox {
public:
    Status initialize(const SandboxConfig& config) {
        (void)config;
        return Error::unsupported("sandbox not implemented in P0.5");
    }

    Status enter() {
        return Error::unsupported("sandbox not implemented in P0.5");
    }

    Status exit() {
        return Error::unsupported("sandbox not implemented in P0.5");
    }

    Status execute(std::string_view command) {
        (void)command;
        return Error::unsupported("sandbox execution not implemented in P0.5");
    }
};

struct LicenseInfo {
    std::string model_id;
    std::string license;
    bool commercial_use{false};
};

// Model license / permission checks (docs/readme.md §20.18).
class ModelSecurity {
public:
    Status verify_license(const std::string& model_path) {
        (void)model_path;
        return Error::unsupported(
            "model license verification not implemented in P0.5");
    }

    Result<LicenseInfo> inspect_license(const std::string& model_path) {
        (void)model_path;
        return Error::unsupported(
            "model license inspection not implemented in P0.5");
    }

    Status check_permission(std::string_view action) {
        (void)action;
        return Error::unsupported(
            "permission checks not implemented in P0.5");
    }
};

struct AuditEntry {
    std::string event;
    std::string actor;
    std::string detail;
    std::int64_t timestamp{0};
};

// Records audit events in memory. Persistence/rotation arrives with the
// deployment milestone; the in-memory behavior is part of the frozen contract.
class AuditLog {
public:
    void record(std::string event) {
        AuditEntry entry;
        entry.event = std::move(event);
        entries_.push_back(std::move(entry));
    }

    void record(AuditEntry entry) { entries_.push_back(std::move(entry)); }

    const std::vector<AuditEntry>& entries() const { return entries_; }

    void clear() { entries_.clear(); }

private:
    std::vector<AuditEntry> entries_;
};

}  // namespace ie
