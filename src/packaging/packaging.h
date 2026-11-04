#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"

namespace ie {

// Distribution metadata and package description (docs/readme.md §20.18).
struct PackageInfo {
    std::string name;
    std::string version;
    std::string format{"tar.gz"};
    std::vector<std::string> contents;
    std::vector<std::string> dependencies;
};

class Packager {
public:
    static Status produce(const PackageInfo& info,
                          const std::string& output_dir) {
        (void)info;
        (void)output_dir;
        return Error::unsupported(
            "packaging not implemented in P0.5");
    }

    static Result<PackageInfo> inspect(const std::string& package_path) {
        (void)package_path;
        return Error::unsupported(
            "package inspection not implemented in P0.5");
    }

    static Status verify(const std::string& package_path) {
        (void)package_path;
        return Error::unsupported(
            "package verification not implemented in P0.5");
    }
};

}  // namespace ie
