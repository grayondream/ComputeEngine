#pragma once

#include <optional>
#include <string>
#include <utility>

namespace ie {

struct Error {
    enum class Code {
        InvalidArgument,
        Unsupported,
        NotFound,
        AlreadyExists,
        OutOfMemory,
        Runtime,
    };

    Code code{Code::Runtime};
    std::string message;

    static Error invalid_argument(std::string message) {
        return {Code::InvalidArgument, std::move(message)};
    }
    static Error unsupported(std::string message) {
        return {Code::Unsupported, std::move(message)};
    }
    static Error not_found(std::string message) {
        return {Code::NotFound, std::move(message)};
    }
    static Error already_exists(std::string message) {
        return {Code::AlreadyExists, std::move(message)};
    }
    static Error out_of_memory(std::string message) {
        return {Code::OutOfMemory, std::move(message)};
    }
    static Error runtime(std::string message) {
        return {Code::Runtime, std::move(message)};
    }
};

// C++20 has no std::expected; this is the minimal expected-like pair used by
// the public API. `Status` is the void specialization.
class Status {
public:
    Status() = default;
    Status(Error error) : error_(std::move(error)) {}

    bool has_value() const { return !error_.has_value(); }
    explicit operator bool() const { return has_value(); }

    Error& error() { return *error_; }
    const Error& error() const { return *error_; }

private:
    std::optional<Error> error_;
};

template <class T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(Error error) : error_(std::move(error)) {}

    bool has_value() const { return value_.has_value(); }
    explicit operator bool() const { return has_value(); }

    T& operator*() { return *value_; }
    const T& operator*() const { return *value_; }
    T* operator->() { return &*value_; }
    const T* operator->() const { return &*value_; }

    T& value() { return *value_; }
    const T& value() const { return *value_; }

    Error& error() { return error_; }
    const Error& error() const { return error_; }

private:
    std::optional<T> value_;
    Error error_{};
};

// Void specialization: success carries no value, mirroring Status.
template <>
class Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)) {}

    bool has_value() const { return !error_.has_value(); }
    explicit operator bool() const { return has_value(); }

    Error& error() { return *error_; }
    const Error& error() const { return *error_; }

private:
    std::optional<Error> error_;
};

}  // namespace ie
