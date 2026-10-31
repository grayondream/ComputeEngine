#pragma once

#include "core/base/error.h"
#include "core/hardware/capability_set.h"
#include "core/hardware/profile.h"

namespace ie {

class HardwareDetector {
public:
    static Result<HardwareProfile> detect_cpu();
    static Result<HardwareProfile> detect_cuda(int device_id);
    static Result<HardwareProfile> detect_vulkan(int device_id);

private:
    static std::string make_fingerprint(const HardwareProfile& profile);
};

}  // namespace ie
