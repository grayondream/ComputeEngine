#pragma once

#include <string>

#include "core/base/error.h"

namespace ie {

struct GrpcConfig {
    std::string address;
    int max_message_bytes{0};
    bool enable_tls{false};
};

// gRPC service surface. The protocol dependency and server are deferred beyond
// P0.5.
class GrpcService {
public:
    Status start(const GrpcConfig& config) {
        (void)config;
        return Error::unsupported("gRPC service not implemented in P0.5");
    }

    Status run() {
        return Error::unsupported("gRPC service not implemented in P0.5");
    }

    Status stop() {
        return Error::unsupported("gRPC service not implemented in P0.5");
    }
};

}  // namespace ie
