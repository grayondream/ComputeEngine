#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct Shape {
    std::vector<int64_t> dims;

    int64_t numel() const;
    int64_t stride(int dim) const;
};

bool operator==(const Shape& lhs, const Shape& rhs);
bool operator!=(const Shape& lhs, const Shape& rhs);
std::string to_string(const Shape& shape);

}  // namespace ie
