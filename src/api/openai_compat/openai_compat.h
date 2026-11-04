#pragma once

#include <string>

#include "core/base/error.h"

namespace ie {

struct OpenAIConfig {
    std::string host{"127.0.0.1"};
    int port{0};
    std::string model;
    int max_concurrent_requests{0};
};

// OpenAI-compatible HTTP endpoint (docs/readme.md §20.18 / §13.6). Declared
// only; serving arrives with the API milestone.
class OpenAIServer {
public:
    Status start(const OpenAIConfig& config) {
        (void)config;
        return Error::unsupported(
            "OpenAI-compatible server not implemented in P0.5");
    }

    Status run() {
        return Error::unsupported(
            "OpenAI-compatible server not implemented in P0.5");
    }

    Status stop() {
        return Error::unsupported(
            "OpenAI-compatible server not implemented in P0.5");
    }
};

}  // namespace ie
