#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"

namespace ie {

// CLI entry point (`ie-run`, `ie-report`, ...; docs/readme.md §13.6). The
// commands are frozen as a surface; argument parsing lands later.
struct CliConfig {
    std::vector<std::string> args;
};

class CliApp {
public:
    Status start() { return Error::unsupported("CLI not implemented in P0.5"); }

    Status run(int argc, char** argv) {
        (void)argc;
        (void)argv;
        return Error::unsupported("CLI not implemented in P0.5");
    }

    Status stop() {
        return Error::unsupported("CLI not implemented in P0.5");
    }
};

}  // namespace ie
