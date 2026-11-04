#pragma once

#include <string>
#include <utility>

#include "core/base/error.h"
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/hardware/profile.h"
#include "core/ir/model_kind.h"
#include "core/ir/op_type.h"
#include "core/tuning/kernel.h"

namespace ie {

// Layer-3 online tuning. P0.5 freezes the contract; tuning itself is deferred.
class RuntimeAutotuner {
public:
    struct TunerConfig {
        bool enabled{true};
        int max_tuning_iterations{3};
        double tuning_budget_ms{100.0};
        bool persist_cache{true};
        std::string cache_dir;
    };

    RuntimeAutotuner() = default;
    explicit RuntimeAutotuner(TunerConfig config)
        : config_(std::move(config)) {}

    const TunerConfig& config() const { return config_; }

    Result<Kernel*> tune(OpType /*op*/, DType /*dtype*/, const Shape& /*shape*/,
                         const HardwareProfile& /*profile*/,
                         ModelKind /*kind*/) {
        return Error::unsupported("runtime autotuning not implemented in P0.5");
    }

private:
    TunerConfig config_;
};

}  // namespace ie
