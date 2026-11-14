#pragma once

#include <string>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "models/traditional/onnx/onnx_wire.h"

namespace ie {

// Loads the supported ONNX subset into the frozen Graph IR. Initializers and
// Constant nodes become Graph::weights; only F32 runtime values are produced.
Result<Graph> load_onnx_model(const std::string& path);

}  // namespace ie
