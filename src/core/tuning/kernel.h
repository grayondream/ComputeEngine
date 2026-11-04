#pragma once

#include <string>

namespace ie {

// Opaque handle for a compiled kernel implementation. P0.5 only freezes the
// surface; concrete backends will subclass this.
class Kernel {
public:
    virtual ~Kernel() = default;

    virtual std::string name() const = 0;
};

}  // namespace ie
