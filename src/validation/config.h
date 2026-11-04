#pragma once

#include <string>

#include "validation/comparator/comparator.h"
#include "validation/determinism/determinism.h"
#include "validation/tensor_hash/tensor_hash.h"

namespace ie {

// Validation configuration (docs/readme.md §10.7). The whole layer is off by
// default and only pays for itself when explicitly enabled.
struct ValidationConfig {
    bool enabled{false};

    struct TracerConfig {
        bool enable_pytorch_trace{false};
        bool enable_onnx_trace{false};
        std::string trace_output_dir;
        bool capture_kv_cache{false};
    } tracer;

    struct HashConfig {
        bool enable{false};
        TensorHasher::HashType hash_type{TensorHasher::HashType::XXH64};
        bool per_tensor_layer{true};
        std::string manifest_path;
    } hasher;

    DeterminismConfig determinism;
    Comparator::CompareConfig comparator;

    struct CrossFrameworkConfig {
        bool enable{false};
        std::string reference_framework;
        std::string reference_output_dir;
    } cross_framework;
};

}  // namespace ie
