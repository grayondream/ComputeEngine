#include "core/base/shape.h"

#include <sstream>

namespace ie {

int64_t Shape::numel() const {
    if (dims.empty()) {
        return 0;
    }
    int64_t total = 1;
    for (int64_t dim : dims) {
        total *= dim;
    }
    return total;
}

int64_t Shape::stride(int dim) const {
    if (dim < 0 || dim >= static_cast<int>(dims.size())) {
        return 0;
    }
    int64_t stride = 1;
    for (int i = dim + 1; i < static_cast<int>(dims.size()); ++i) {
        stride *= dims[i];
    }
    return stride;
}

bool operator==(const Shape& lhs, const Shape& rhs) {
    return lhs.dims == rhs.dims;
}

bool operator!=(const Shape& lhs, const Shape& rhs) {
    return !(lhs == rhs);
}

std::string to_string(const Shape& shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << shape.dims[i];
    }
    out << ']';
    return out.str();
}

}  // namespace ie
