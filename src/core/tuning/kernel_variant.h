#pragma once

#include <string>

#include "core/base/types.h"
#include "core/hardware/capability_set.h"
#include "core/ir/op_type.h"

namespace ie {

struct KernelVariant {
    std::string name;
    OpType op{OpType::Add};
    DType dtype{DType::F32};
    CapabilitySet required;
};

}  // namespace ie
