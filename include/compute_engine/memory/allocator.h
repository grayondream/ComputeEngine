#pragma once

#include "compute_engine/backend/backend.h"
#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include "compute_engine/core/dtype.h"
#include "compute_engine/core/shape.h"
#include "compute_engine/core/tensor.h"
#include "compute_engine/core/device_type.h"
#include <cstddef>
#include <memory>

namespace compute_engine {

class CPUAllocator : public Allocator {
public:
    Status Allocate(void** ptr, size_t size) override;
    Status Deallocate(void* ptr) override;
    Status Copy(void* dst, const void* src,
                size_t size, CopyDirection dir) override;
    Status Memset(void* ptr, int value, size_t size) override;
    Status CreateTensor(TensorPtr& tensor,
                        const Shape& shape, DType dtype) override;
};

} // namespace compute_engine
