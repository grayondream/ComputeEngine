#pragma once

#include <string>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/tuning/kernel_variant.h"

namespace ie {

// Persistent latency cache keyed by hardware + op + dtype + shape + model kind.
// P0.5 freezes the shape of the cache; persistence is not implemented yet.
class HardwareAwareCache {
public:
    struct Key {
        std::string hardware_fingerprint;
        OpType op{OpType::Add};
        DType dtype{DType::F32};
        Shape shape;
        ModelKind kind{ModelKind::UNKNOWN};
    };

    struct Value {
        KernelVariant variant;
        double latency_us{0.0};
    };

    Result<Value> get(const Key& /*key*/) {
        return Error::unsupported(
            "hardware-aware cache not implemented in P0.5");
    }

    Status put(const Key& /*key*/, const Value& /*value*/) {
        return Error::unsupported(
            "hardware-aware cache not implemented in P0.5");
    }

    Status save(const std::string& /*path*/) {
        return Error::unsupported(
            "hardware-aware cache not implemented in P0.5");
    }

    Status load(const std::string& /*path*/) {
        return Error::unsupported(
            "hardware-aware cache not implemented in P0.5");
    }
};

}  // namespace ie
