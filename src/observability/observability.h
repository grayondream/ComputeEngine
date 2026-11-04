#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/backend/backend_caps.h"
#include "core/base/error.h"
#include "core/base/types.h"
#include "core/config/feature_flags.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/ir/model_kind.h"
#include "hybrid/hybrid_config.h"

namespace ie {

// Trace event collection (docs/readme.md §13.1). P0.5 keeps the event methods
// as no-ops and reports Unsupported when asked to persist a trace.
class TraceExporter {
public:
    void begin_event(const std::string& name, const std::string& category) {
        (void)name;
        (void)category;
    }

    void end_event() {}

    void instant_event(const std::string& name, const std::string& category) {
        (void)name;
        (void)category;
    }

    void counter(const std::string& name, double value) {
        (void)name;
        (void)value;
    }

    void set_device(Device device) { (void)device; }
    void set_stream(int stream_id) { (void)stream_id; }
    void set_phase(const std::string& phase) { (void)phase; }
    void set_model_kind(ModelKind kind) { (void)kind; }

    Status flush(const std::string& path) {
        (void)path;
        return Error::unsupported("trace export not implemented in P0.5");
    }
};

// Graph serialization (DOT/JSON/Mermaid, §13.1). Deferred with JSON/export
// work in P0.5.
class GraphExporter {
public:
    static Status to_dot(const Graph& graph, const std::string& path) {
        (void)graph;
        (void)path;
        return Error::unsupported("graph export not implemented in P0.5");
    }

    static Status to_json(const Graph& graph, const std::string& path) {
        (void)graph;
        (void)path;
        return Error::unsupported("graph export not implemented in P0.5");
    }

    static Status to_mermaid(const Graph& graph, const std::string& path) {
        (void)graph;
        (void)path;
        return Error::unsupported("graph export not implemented in P0.5");
    }
};

class MetricsExporter {
public:
    Status to_prometheus(const std::string& path) {
        (void)path;
        return Error::unsupported("metrics export not implemented in P0.5");
    }

    Status to_json(const std::string& path) {
        (void)path;
        return Error::unsupported("metrics export not implemented in P0.5");
    }
};

// Reproducible experiment record (docs/readme.md §13.3).
struct Experiment {
    std::string id;
    std::string name;

    struct Config {
        std::vector<BackendKind> devices;
        HybridConfig hybrid;
        std::vector<std::string> passes;
        std::string quant_format;
        std::string scheduler_policy;
        std::string kv_cache_policy;
        ModelKind model_kind{ModelKind::UNKNOWN};
        bool is_training{false};
        std::string optimizer;
        std::string lr_scheduler;
        std::string loss;
        bool use_amp{false};
        bool enable_validation{false};
        bool enable_deterministic{false};
        FeatureFlags features;
        std::unordered_map<std::string, std::string> extras;
    } config;

    std::string graph_path;
    std::string input_path;
    HardwareProfile hardware_profile;

    struct Result {
        double latency_p50{0.0};
        double latency_p95{0.0};
        double latency_p99{0.0};
        double throughput{0.0};
        std::size_t memory_peak{0};
        double accuracy{0.0};
        double final_loss{0.0};
        double best_metric{0.0};
        int total_steps{0};
        bool validation_passed{false};
        double max_divergence{0.0};
        std::string trace_path;
        std::string metrics_path;
    } result;
};

struct Diff {
    std::string field;
    double baseline{0.0};
    double candidate{0.0};
};

class ExperimentStore {
public:
    void save(const Experiment& experiment) { (void)experiment; }

    Result<Experiment> load(const std::string& id) {
        (void)id;
        return Error::unsupported(
            "experiment store not implemented in P0.5");
    }

    std::vector<Experiment> list(const std::string& filter) {
        (void)filter;
        return {};
    }

    Status export_csv(const std::string& path) {
        (void)path;
        return Error::unsupported(
            "experiment export not implemented in P0.5");
    }

    Status export_json(const std::string& path) {
        (void)path;
        return Error::unsupported(
            "experiment export not implemented in P0.5");
    }

    std::vector<Experiment> find_by_capability(const std::string& cap_name,
                                               bool value) {
        (void)cap_name;
        (void)value;
        return {};
    }

    std::vector<Experiment> find_by_bandwidth_range(float min, float max) {
        (void)min;
        (void)max;
        return {};
    }

    std::vector<Experiment> find_by_model_kind(ModelKind kind) {
        (void)kind;
        return {};
    }
};

class ExperimentRunner {
public:
    struct Matrix {
        std::vector<std::vector<BackendKind>> device_sets = {
            {BackendKind::CPU},
            {BackendKind::CPU, BackendKind::CUDA},
            {BackendKind::CPU, BackendKind::VULKAN},
        };
        std::vector<HybridConfig::Strategy> strategies;
        std::vector<std::string> quant_formats;
        std::vector<std::string> scheduler_policies;
        std::vector<std::string> kv_cache_policies;
        std::vector<std::string> optimizers;
        std::vector<std::string> lr_schedulers;
        std::vector<ModelKind> model_kinds;
    };

    Result<std::vector<Experiment>> run_matrix(const Matrix& matrix,
                                               const Graph& graph) {
        (void)matrix;
        (void)graph;
        return Error::unsupported(
            "experiment matrix not implemented in P0.5");
    }

    Result<std::vector<Experiment>> run_ablation(
        const Experiment::Config& base, const std::string& variable,
        const std::vector<std::string>& values) {
        (void)base;
        (void)variable;
        (void)values;
        return Error::unsupported(
            "experiment ablation not implemented in P0.5");
    }

    Result<std::vector<Experiment>> run_cross_hardware(
        const std::vector<HardwareProfile>& profiles, const Graph& graph) {
        (void)profiles;
        (void)graph;
        return Error::unsupported(
            "cross-hardware comparison not implemented in P0.5");
    }
};

class ComparisonReport {
public:
    static Status to_markdown(const std::vector<Experiment>& experiments,
                              const std::string& path) {
        (void)experiments;
        (void)path;
        return Error::unsupported(
            "comparison report not implemented in P0.5");
    }

    static Status to_html(const std::vector<Experiment>& experiments,
                          const std::string& path) {
        (void)experiments;
        (void)path;
        return Error::unsupported(
            "comparison report not implemented in P0.5");
    }

    static Status to_csv(const std::vector<Experiment>& experiments,
                         const std::string& path) {
        (void)experiments;
        (void)path;
        return Error::unsupported(
            "comparison report not implemented in P0.5");
    }

    static std::vector<Diff> compare(const Experiment& baseline,
                                     const Experiment& candidate) {
        (void)baseline;
        (void)candidate;
        return {};
    }

    static Status to_cross_hardware_report(
        const std::vector<Experiment>& experiments, const std::string& path) {
        (void)experiments;
        (void)path;
        return Error::unsupported(
            "cross-hardware report not implemented in P0.5");
    }
};

struct Metric {
    std::string name;
    double value{0.0};
    std::unordered_map<std::string, std::string> labels;
};

// In-process metrics registry. Collection is deferred; the surface is frozen
// so exporters can target it later.
class MetricsRegistry {
public:
    void clear() {}

    void increment(std::string_view name, double delta = 1.0) {
        (void)name;
        (void)delta;
    }

    void observe(std::string_view name, double value) {
        (void)name;
        (void)value;
    }

    std::vector<Metric> snapshot() const { return {}; }
};

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

// Minimal logging facade. The default logger writes nothing in P0.5.
class Logger {
public:
    void set_level(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    void log(LogLevel level, std::string_view message) {
        (void)level;
        (void)message;
    }

private:
    LogLevel level_{LogLevel::INFO};
};

struct TraceSpan {
    std::string name;
    std::string category;
    double start_us{0.0};
    double duration_us{0.0};
};

class Tracer {
public:
    void begin_span(const std::string& name, const std::string& category) {
        (void)name;
        (void)category;
    }

    void end_span() {}

    std::vector<TraceSpan> spans() const { return {}; }
};

struct ProfileSample {
    std::string name;
    double duration_us{0.0};
};

class Profiler {
public:
    void begin(const std::string& name) { (void)name; }
    void end() {}
    void clear() {}

    std::vector<ProfileSample> samples() const { return {}; }
};

}  // namespace ie
