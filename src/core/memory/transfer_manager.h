#pragma once

#include <cstddef>

#include "core/base/error.h"
#include "core/base/types.h"
#include "core/memory/allocator.h"

namespace ie {

// P0 skeleton: only same-device copies are accepted.
class TransferManager {
public:
    Status copy(Device src, Device dst, BufferHandle /*src_handle*/,
                std::size_t /*bytes*/) {
        if (src == dst) {
            return {};
        }
        return Error::unsupported("cross-device copy not implemented in P0");
    }
};

}  // namespace ie
