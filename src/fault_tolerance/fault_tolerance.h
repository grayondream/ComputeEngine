#pragma once

#include <string>
#include <string_view>

#include "core/base/error.h"

namespace ie {

enum class HealthState { HEALTHY, DEGRADED, UNHEALTHY };

struct HealthStatus {
    HealthState state{HealthState::HEALTHY};
    std::string summary;
};

// Process/component health aggregation. Monitoring is not wired up in P0.5.
class HealthMonitor {
public:
    Status start() { return {}; }
    Status stop() { return {}; }

    HealthStatus status() const { return {}; }

    Status report_failure(std::string_view component, std::string_view message) {
        (void)component;
        (void)message;
        return Error::unsupported(
            "health failure reporting not implemented in P0.5");
    }
};

struct ResilienceConfig {
    int max_retries{3};
    int retry_backoff_ms{0};
    int timeout_ms{0};
    bool enable_circuit_breaker{false};
};

struct ResilienceDecision {
    bool retry{false};
    bool open_circuit{false};
    std::string reason;
};

// Retry/timeout/circuit-breaker policy contract. Evaluation is deferred.
class ResiliencePolicy {
public:
    Status configure(const ResilienceConfig& config) {
        config_ = config;
        return {};
    }

    const ResilienceConfig& config() const { return config_; }

    Status reset() { return {}; }

    Result<ResilienceDecision> evaluate(std::string_view component,
                                        int attempt) {
        (void)component;
        (void)attempt;
        return Error::unsupported(
            "resilience policy not implemented in P0.5");
    }

private:
    ResilienceConfig config_{};
};

}  // namespace ie
