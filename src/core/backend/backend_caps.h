#pragma once

#include <cstddef>

namespace ie {

enum class BackendKind { CPU, CUDA, VULKAN };

// Backend capability declaration (docs/readme.md §3.7).
struct BackendCaps {
    bool supports_async{false};
    bool supports_unified_memory{false};
    bool supports_graph_capture{false};

    // Control flow
    bool supports_control_flow{false};

    // Training reservation
    bool supports_training{false};
    bool supports_amp{false};
    bool supports_gradient_checkpointing{false};
    std::size_t max_grad_buffer_size{0};

    // Validation reservation
    bool supports_deterministic_kernels{false};
    bool supports_batch_invariant{false};

    // Hardware adaptation
    bool supports_runtime_autotuning{false};
    bool supports_persistent_tuning_cache{false};
};

}  // namespace ie
