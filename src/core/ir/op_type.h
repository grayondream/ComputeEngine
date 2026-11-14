#pragma once

namespace ie {

// Mirrors docs/readme.md §4.2. P0 implements only Add/Mul/MatMul runtimes;
// the rest are reserved op codes.
enum class OpType {
    // Generic tensor ops
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Relu,
    Softmax,
    Concat,
    Slice,
    Reshape,
    Transpose,
    Gather,
    Scatter,
    Reduce,

    // Traditional model ops
    Conv2D,
    DepthwiseConv2D,
    ConvTranspose2D,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool,
    BatchNorm,
    LayerNorm,
    GroupNorm,
    LSTM,
    GRU,
    RNN,
    Upsample,
    Pad,
    Clip,
    Sigmoid,
    Tanh,
    GBDT_Split,
    GBDT_Leaf,

    // Transformer ops
    Attention,
    MultiHeadAttention,
    RoPE,
    RMSNorm,
    Embedding,
    PositionalEncoding,
    MoE_Router,
    MoE_Expert,

    // Control flow
    If,
    Loop,
    While,
    Subgraph,

    // Rule ops
    RuleMatch,
    RuleFire,
    FactAssert,

    // Orchestration ops
    ToolCall,
    LLMGenerate,
    Route,
    Merge,

    // Training-reserved ops
    CrossEntropy,
    MSE,
    Dropout,
    SoftmaxCrossEntropy,

    // Additional ops appended to keep existing numeric values stable.
    Flatten,
};

}  // namespace ie
