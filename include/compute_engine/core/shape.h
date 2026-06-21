#pragma once

#include "compute_engine/core/macros.h"
#include "compute_engine/core/status.h"
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace compute_engine {

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    size_t ndim() const { return dims_.size(); }

    int64_t num_elements() const {
        if (dims_.empty()) return 0;
        int64_t n = 1;
        for (auto d : dims_) n *= d;
        return n;
    }

    int64_t operator[](size_t i) const { return dims_[i]; }
    int64_t& operator[](size_t i) { return dims_[i]; }

    int64_t at(size_t i) const { return dims_.at(i); }

    Status Reshape(const Shape& new_shape) {
        if (new_shape.num_elements() != num_elements()) {
            return Status::InvalidArgument("Reshape: element count mismatch");
        }
        dims_ = new_shape.dims_;
        return Status::OK();
    }

    Shape Broadcast(const Shape& other) const {
        size_t max_ndim = std::max(ndim(), other.ndim());
        std::vector<int64_t> result(max_ndim);
        for (size_t i = 0; i < max_ndim; ++i) {
            int64_t d1 = (i < ndim()) ? dims_[ndim() - 1 - i] : 1;
            int64_t d2 = (i < other.ndim()) ? other.dims_[other.ndim() - 1 - i] : 1;
            if (d1 == 1) result[max_ndim - 1 - i] = d2;
            else if (d2 == 1) result[max_ndim - 1 - i] = d1;
            else if (d1 == d2) result[max_ndim - 1 - i] = d1;
            else result[max_ndim - 1 - i] = 0; // error
        }
        return Shape(result);
    }

    std::string ToString() const {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << dims_[i];
        }
        ss << "]";
        return ss.str();
    }

    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }

    bool operator==(const Shape& o) const { return dims_ == o.dims_; }
    bool operator!=(const Shape& o) const { return !(*this == o); }

    const std::vector<int64_t>& dims() const { return dims_; }

private:
    std::vector<int64_t> dims_;
};

} // namespace compute_engine
