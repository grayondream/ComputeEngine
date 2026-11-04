#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"

namespace ie {

// Benchmark definition from docs/readme.md §20.16. A case always names the
// model/graph it measures; iterations/warmup are optional knobs.
struct BenchmarkCase {
    std::string name;
    std::string graph_path;
    int iterations{1};
    int warmup{0};
};

struct BenchmarkResult {
    std::string name;
    double latency_p50{0.0};
    double latency_p95{0.0};
    double latency_p99{0.0};
    double throughput{0.0};
    bool passed{true};
};

class BenchmarkRunner {
public:
    Result<BenchmarkResult> run(const BenchmarkCase& benchmark_case) {
        (void)benchmark_case;
        return Error::unsupported(
            "benchmark runner not implemented in P0.5");
    }

    Status run_all(const std::vector<BenchmarkCase>& cases,
                   const std::string& output_path) {
        (void)cases;
        (void)output_path;
        return Error::unsupported(
            "benchmark runner not implemented in P0.5");
    }
};

}  // namespace ie
