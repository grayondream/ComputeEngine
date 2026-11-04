#pragma once

#include "core/ops/unimplemented_op.h"

namespace ie {

// Traditional model ops (docs/readme.md §3.6). P0.5 freezes the surface only.

class Conv2DOp final : public UnimplementedOp {
public:
    Conv2DOp() : UnimplementedOp("Conv2D") {}
};

class DepthwiseConv2DOp final : public UnimplementedOp {
public:
    DepthwiseConv2DOp() : UnimplementedOp("DepthwiseConv2D") {}
};

class ConvTranspose2DOp final : public UnimplementedOp {
public:
    ConvTranspose2DOp() : UnimplementedOp("ConvTranspose2D") {}
};

class MaxPool2DOp final : public UnimplementedOp {
public:
    MaxPool2DOp() : UnimplementedOp("MaxPool2D") {}
};

class AvgPool2DOp final : public UnimplementedOp {
public:
    AvgPool2DOp() : UnimplementedOp("AvgPool2D") {}
};

class GlobalAvgPoolOp final : public UnimplementedOp {
public:
    GlobalAvgPoolOp() : UnimplementedOp("GlobalAvgPool") {}
};

class BatchNormOp final : public UnimplementedOp {
public:
    BatchNormOp() : UnimplementedOp("BatchNorm") {}
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
