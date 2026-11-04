#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/ir/attr.h"

namespace ie {

struct ModelMeta {
    std::string name;
    std::string framework;
    std::string version;
};

// Framework-neutral reference trace (docs/readme.md §10.2). It deliberately
// does not depend on PyTorch or ONNX types.
struct TraceIR {
    struct TensorRecord {
        std::string name;
        Shape shape;
        DType dtype{DType::F32};
        std::vector<std::uint8_t> data;
        std::string op_type;
        std::int64_t timestamp{0};
    };

    struct OpRecord {
        std::string op_name;
        std::string op_type;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        Attrs attrs;
    };

    std::vector<TensorRecord> tensors;
    std::vector<OpRecord> ops;
    ModelMeta model_meta;
};

}  // namespace ie
