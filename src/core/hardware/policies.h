#pragma once

#include "core/ir/op_type.h"

namespace ie {

struct QuantPolicy {
    enum class Mode { NONE, INT8, FP8 };
    Mode mode{Mode::NONE};
};

}  // namespace ie
