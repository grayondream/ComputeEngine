#pragma once

#include <string>

#include "core/base/error.h"
#include "core/hardware/profile.h"
#include "core/ops/kernel_fns.h"

namespace ie {

// Maps the winning capability-driven variants to concrete F32 kernels.
class CpuKernelCatalog {
public:
    struct Selection {
        std::string matmul;
        std::string relu;
        std::string softmax;
        KernelFns fns;
    };

    // Selects MatMul/Relu/Softmax variants for the profile. Reuses one static
    // variant registry; every returned function pointer is non-null.
    static Result<Selection> select(const HardwareProfile& profile);
};

}  // namespace ie
