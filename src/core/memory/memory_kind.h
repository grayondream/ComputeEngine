#pragma once

namespace ie {

enum class MemoryKind {
    HOST,
    DEVICE,
    PINNED,
    UNIFIED,
    MANAGED,
    GRADIENT,
    OPTIMIZER_STATE,
    ACTIVATION,
    // Traditional-model specific
    WEIGHT_CONST,
    // Transformer specific
    KV_CACHE,
};

}  // namespace ie
