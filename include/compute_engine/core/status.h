#pragma once

#include "compute_engine/core/macros.h"
#include <string>
#include <cstdint>

namespace compute_engine {

enum class ErrorCode : uint8_t {
    OK = 0,
    RuntimeError,
    InvalidArgument,
    NotImplemented,
    InternalError,
    OutOfRange,
    InvalidState,
};

class Status {
public:
    Status() : code_(ErrorCode::OK) {}

    static Status OK() { return Status(); }

    static Status RuntimeError(const std::string& msg) {
        Status s; s.code_ = ErrorCode::RuntimeError; s.msg_ = msg; return s;
    }

    static Status InvalidArgument(const std::string& msg) {
        Status s; s.code_ = ErrorCode::InvalidArgument; s.msg_ = msg; return s;
    }

    static Status NotImplemented(const std::string& msg) {
        Status s; s.code_ = ErrorCode::NotImplemented; s.msg_ = msg; return s;
    }

    static Status InternalError(const std::string& msg) {
        Status s; s.code_ = ErrorCode::InternalError; s.msg_ = msg; return s;
    }

    static Status OutOfRange(const std::string& msg) {
        Status s; s.code_ = ErrorCode::OutOfRange; s.msg_ = msg; return s;
    }

    static Status InvalidState(const std::string& msg) {
        Status s; s.code_ = ErrorCode::InvalidState; s.msg_ = msg; return s;
    }

    static Status NotFound(const std::string& msg) {
        Status s; s.code_ = ErrorCode::RuntimeError; s.msg_ = msg; return s;
    }

    bool ok() const { return code_ == ErrorCode::OK; }
    ErrorCode code() const { return code_; }
    const std::string& message() const { return msg_; }

    std::string ToString() const;

    bool operator==(const Status& o) const { return code_ == o.code_; }
    bool operator!=(const Status& o) const { return !(*this == o); }

    Status(const Status&) = default;
    Status& operator=(const Status&) = default;
    Status(Status&&) = default;
    Status& operator=(Status&&) = default;

private:
    ErrorCode code_;
    std::string msg_;
};

inline Status operator"" _ok(unsigned long long) {
    return Status::OK();
}

} // namespace compute_engine
