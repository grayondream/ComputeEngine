#pragma once

#include "core/base/types.h"

namespace ie {

struct Stream {
    Device device{Device::CPU};
    void* handle{nullptr};
};

}  // namespace ie
