#pragma once

#include <string>

#include "core/base/error.h"

namespace ie {

struct ValidationReport {
    bool passed{false};
    std::string summary;
};

// Serializes validation reports. Not implemented in P0.5.
class ReportGenerator {
public:
    static Status to_json(const ValidationReport& report,
                          const std::string& path) {
        (void)report;
        (void)path;
        return Error::unsupported("report generator not implemented in P0.5");
    }

    static Status to_markdown(const ValidationReport& report,
                              const std::string& path) {
        (void)report;
        (void)path;
        return Error::unsupported("report generator not implemented in P0.5");
    }
};

}  // namespace ie
