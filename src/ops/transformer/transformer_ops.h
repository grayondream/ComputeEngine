#pragma once

#include "core/ops/unimplemented_op.h"

namespace ie {

// Transformer model ops (docs/readme.md §5). P0.5 freezes the surface only.

class AttentionOp final : public UnimplementedOp {
public:
    AttentionOp() : UnimplementedOp("Attention") {}
};

class MultiHeadAttentionOp final : public UnimplementedOp {
public:
    MultiHeadAttentionOp() : UnimplementedOp("MultiHeadAttention") {}
};

class RoPEOp final : public UnimplementedOp {
public:
    RoPEOp() : UnimplementedOp("RoPE") {}
};

class RMSNormOp final : public UnimplementedOp {
public:
    RMSNormOp() : UnimplementedOp("RMSNorm") {}
};

class EmbeddingOp final : public UnimplementedOp {
public:
    EmbeddingOp() : UnimplementedOp("Embedding") {}
};

class PositionalEncodingOp final : public UnimplementedOp {
public:
    PositionalEncodingOp() : UnimplementedOp("PositionalEncoding") {}
};

class MoERouterOp final : public UnimplementedOp {
public:
    MoERouterOp() : UnimplementedOp("MoERouter") {}
};

class MoEExpertOp final : public UnimplementedOp {
public:
    MoEExpertOp() : UnimplementedOp("MoEExpert") {}
};

}  // namespace ie
