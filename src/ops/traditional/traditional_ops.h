#pragma once

#include "core/ops/unimplemented_op.h"
#include "ops/traditional/batch_norm_op.h"
#include "ops/traditional/conv2d_op.h"
#include "ops/traditional/gemm_op.h"
#include "ops/traditional/layout_ops.h"
#include "ops/traditional/pool_ops.h"

namespace ie {

// Traditional model ops (docs/readme.md §3.6). Conv2D, DepthwiseConv2D,
// MaxPool2D, AvgPool2D, GlobalAvgPool, BatchNorm, Gemm, Flatten, Reshape, and
// Transpose are implemented in their own headers; the classes below remain P2b
// stubs.

class ConvTranspose2DOp final : public UnimplementedOp {
public:
    ConvTranspose2DOp() : UnimplementedOp("ConvTranspose2D") {}
};

class LayerNormOp final : public UnimplementedOp {
public:
    LayerNormOp() : UnimplementedOp("LayerNorm") {}
};

class GroupNormOp final : public UnimplementedOp {
public:
    GroupNormOp() : UnimplementedOp("GroupNorm") {}
};

class LSTMOp final : public UnimplementedOp {
public:
    LSTMOp() : UnimplementedOp("LSTM") {}
};

class GRUOp final : public UnimplementedOp {
public:
    GRUOp() : UnimplementedOp("GRU") {}
};

class RNNOp final : public UnimplementedOp {
public:
    RNNOp() : UnimplementedOp("RNN") {}
};

class UpsampleOp final : public UnimplementedOp {
public:
    UpsampleOp() : UnimplementedOp("Upsample") {}
};

class PadOp final : public UnimplementedOp {
public:
    PadOp() : UnimplementedOp("Pad") {}
};

class ClipOp final : public UnimplementedOp {
public:
    ClipOp() : UnimplementedOp("Clip") {}
};

class SigmoidOp final : public UnimplementedOp {
public:
    SigmoidOp() : UnimplementedOp("Sigmoid") {}
};

class TanhOp final : public UnimplementedOp {
public:
    TanhOp() : UnimplementedOp("Tanh") {}
};

class GBDTSplitOp final : public UnimplementedOp {
public:
    GBDTSplitOp() : UnimplementedOp("GBDTSplit") {}
};

class GBDTLeafOp final : public UnimplementedOp {
public:
    GBDTLeafOp() : UnimplementedOp("GBDTLeaf") {}
};

}  // namespace ie
