#pragma once

#include <string>

#include "core/base/error.h"
#include "core/hardware/profile.h"
#include "core/ir/graph.h"
#include "core/ir/model_kind.h"

namespace ie {

// Shared state handed to every compiler pass. Hardware comes from the probe
// (or a stub), and model_kind lets kind-specific passes filter themselves.
struct PassContext {
    HardwareProfile hardware;
    ModelKind model_kind{ModelKind::UNKNOWN};
};

class Pass {
public:
    virtual ~Pass() = default;

    virtual std::string name() const = 0;
    virtual Status run(Graph& graph, PassContext& context) = 0;
};

}  // namespace ie
