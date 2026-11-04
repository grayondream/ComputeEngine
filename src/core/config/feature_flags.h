#pragma once

namespace ie {

// Runtime feature gates (docs/readme.md §12.3). Optional, experimental, and
// training paths stay disabled by default; the core inference path stays on.
struct FeatureFlags {
    bool enable_experimental = false;
    bool enable_fp8 = false;
    bool enable_speculative = false;
    bool enable_paged_attention = false;
    bool enable_moe = false;
    bool enable_training = false;
    bool enable_amp = false;
    bool enable_distributed = false;
    bool enable_validation = false;
    bool enable_deterministic = false;
    bool enable_reference_trace = false;
    bool enable_capability_driven_selection = true;
    bool enable_runtime_autotuning = true;
    // ===== Traditional models =====
    bool enable_conv_winograd = true;
    bool enable_conv_fft = false;
    bool enable_int8_ptq = true;
    // ===== Transformer =====
    bool enable_flash_attention = true;
    bool enable_kv_cache_quant = true;
};

// Process-wide feature gates. Prefer passing explicit flags through
// EngineConfig in new code; this accessor is for global switches.
inline FeatureFlags& global_feature_flags() {
    static FeatureFlags flags;
    return flags;
}

}  // namespace ie
