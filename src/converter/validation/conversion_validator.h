#pragma once

#include <string>
#include <vector>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "core/memory/tensor.h"

namespace ie {

// Five-stage conversion validation contract (docs/readme.md §11.3). P0.5
// freezes the shape of the API; validation itself is unsupported.
class ConversionValidator {
public:
    struct ValidationConfig {
        double rtol{1e-3};
        double atol{1e-5};
        bool layer_by_layer{true};
        bool check_tokenizer{true};
        bool check_metadata{true};
    };

    struct ValidationResult {
        bool passed{false};
        std::string stage;
        std::string detail;
    };

    static Result<ValidationResult> validate_structure(
        const Graph& /*source*/, const Graph& /*target*/) {
        return Error::unsupported(
            "structure validation not implemented in P0.5");
    }

    static Result<ValidationResult> validate_weights(
        const std::string& /*source_path*/,
        const std::string& /*target_path*/) {
        return Error::unsupported("weight validation not implemented in P0.5");
    }

    static Result<ValidationResult> validate_layer_outputs(
        const std::string& /*source_path*/, const std::string& /*target_path*/,
        const std::vector<Tensor>& /*inputs*/) {
        return Error::unsupported(
            "layer output validation not implemented in P0.5");
    }

    static Result<ValidationResult> validate_end_to_end(
        const std::string& /*source_path*/, const std::string& /*target_path*/,
        const std::vector<Tensor>& /*inputs*/) {
        return Error::unsupported(
            "end-to-end validation not implemented in P0.5");
    }

    static Result<ValidationResult> validate_tokenizer_metadata(
        const std::string& /*source_path*/,
        const std::string& /*target_path*/) {
        return Error::unsupported(
            "tokenizer metadata validation not implemented in P0.5");
    }
};

}  // namespace ie
