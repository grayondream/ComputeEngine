#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/base/error.h"
#include "core/plugin/plugin_meta.h"

namespace ie {

// Tokenizer contract (docs/readme.md §8.1). P0.5 freezes the surface only;
// concrete tokenizers arrive with the LLM milestone.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual Result<std::vector<int>> encode(std::string_view text) = 0;
    virtual Result<std::string> decode(std::span<const int> tokens) = 0;
};

struct TokenizerConfig {
    std::string path;
};

class TokenizerPlugin {
public:
    virtual ~TokenizerPlugin() = default;

    virtual PluginMeta meta() const = 0;
    virtual std::unique_ptr<Tokenizer> create(const TokenizerConfig& config) = 0;
};

}  // namespace ie
