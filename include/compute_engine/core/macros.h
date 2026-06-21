#pragma once

#include <cstdint>

#ifdef __GNUC__
#define CE_UNUSED(x) ((void)(x))
#define CE_LIKELY(x) __builtin_expect(!!(x), 1)
#define CE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define CE_UNUSED(x) ((void)(x))
#define CE_LIKELY(x) (x)
#define CE_UNLIKELY(x) (x)
#else
#define CE_UNUSED(x) ((void)(x))
#define CE_LIKELY(x) (x)
#define CE_UNLIKELY(x) (x)
#endif

#ifdef _WIN32
#define CE_EXPORT __declspec(dllexport)
#define CE_IMPORT __declspec(dllimport)
#else
#define CE_EXPORT __attribute__((visibility("default")))
#define CE_IMPORT
#endif

#ifdef CE_EXPORT_SYMBOLS
#define CE_API CE_EXPORT
#else
#define CE_API CE_IMPORT
#endif

#define CE_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&) = delete;        \
    TypeName& operator=(const TypeName&) = delete;

#define CE_DISALLOW_COPY_ASSIGN_AND_MOVE(TypeName) \
    TypeName(const TypeName&) = delete;             \
    TypeName& operator=(const TypeName&) = delete;  \
    TypeName(TypeName&&) = delete;                  \
    TypeName& operator=(TypeName&&) = delete;
