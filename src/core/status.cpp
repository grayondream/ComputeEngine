#include "compute_engine/core/status.h"
#include "compute_engine/platform/logger.h"

namespace compute_engine {

std::string Status::ToString() const {
    if (ok()) return "OK";
    std::string result;
    switch (code_) {
        case ErrorCode::RuntimeError: result = "RuntimeError"; break;
        case ErrorCode::InvalidArgument: result = "InvalidArgument"; break;
        case ErrorCode::NotImplemented: result = "NotImplemented"; break;
        case ErrorCode::InternalError: result = "InternalError"; break;
        case ErrorCode::OutOfRange: result = "OutOfRange"; break;
        case ErrorCode::InvalidState: result = "InvalidState"; break;
        default: result = "Unknown"; break;
    }
    if (!msg_.empty()) result += ": " + msg_;
    return result;
}

} // namespace compute_engine
