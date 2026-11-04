#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/memory/tensor.h"
#include "validation/reference_tracer/trace_ir.h"

namespace ie {

struct TensorDiff {
    bool passed{false};
    double max_abs_diff{0.0};
    double max_rel_diff{0.0};
    std::size_t mismatch_count{0};
};

struct DivergencePoint {
    std::string tensor_name;
    std::size_t index{0};
    double reference{0.0};
    double candidate{0.0};
};

struct CompareReport {
    bool passed{false};
    std::vector<TensorDiff> tensor_diffs;
    std::vector<DivergencePoint> divergences;
    std::string summary;
};

// Tensor-level comparator (docs/readme.md §10.5). Not implemented in P0.5:
// every entry point reports Unsupported.
class Comparator {
public:
    struct CompareConfig {
        double rtol{1e-3};
        double atol{1e-5};
        bool compare_first_token{true};
        int max_divergence_points{10};
        bool compare_kv_cache{true};
    };

    static Result<CompareReport> compare(const TraceIR& reference,
                                         const TraceIR& candidate,
                                         const CompareConfig& config) {
        (void)reference;
        (void)candidate;
        (void)config;
        return Error::unsupported("comparator not implemented in P0.5");
    }

    static Result<TensorDiff> compare_tensor(const Tensor& ref,
                                             const Tensor& cand,
                                             const CompareConfig& config) {
        (void)ref;
        (void)cand;
        (void)config;
        return Error::unsupported("comparator not implemented in P0.5");
    }

    static Result<DivergencePoint> find_first_divergence(
        const TraceIR& reference, const TraceIR& candidate) {
        (void)reference;
        (void)candidate;
        return Error::unsupported("comparator not implemented in P0.5");
    }
};

}  // namespace ie
