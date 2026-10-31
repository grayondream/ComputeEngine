#pragma once

namespace ie {

enum class ModelKind {
    TRADITIONAL,  // CNN / RNN / MLP / GBDT
    TRANSFORMER,  // LLM / ViT / BERT
    HYBRID,       // e.g. CNN + Transformer
    RULE,         // rule inference
    UNKNOWN,
};

}  // namespace ie
