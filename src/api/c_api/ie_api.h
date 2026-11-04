#pragma once

#include "core/backend/backend.h"
#include "core/plugin/plugin_api.h"

// Frozen C ABI version (docs/readme.md §12.4). Keep the macros in sync with
// the C++ constants above; the static assertions below enforce that at compile
// time.
#define IE_API_VERSION_MAJOR 0
#define IE_API_VERSION_MINOR 1

#if defined(_WIN32)
#define IE_API __declspec(dllexport)
#else
#define IE_API __attribute__((visibility("default")))
#endif

namespace ie {
namespace detail {

static_assert(IE_API_VERSION_MAJOR == kApiVersionMajor,
              "C ABI major version must match kApiVersionMajor");
static_assert(IE_API_VERSION_MINOR == kApiVersionMinor,
              "C ABI minor version must match kApiVersionMinor");
static_assert(kApiVersion == kApiVersionString,
              "kApiVersion and kApiVersionString must match");

}  // namespace detail
}  // namespace ie

extern "C" {
// Creates and initializes a backend. P0.5 recognises only "cpu"; unknown or
// null kinds return nullptr. The optional config string is reserved for future
// backend-specific options.
IE_API ie::Backend* ie_create_backend(const char* kind, const char* config);
// Destroys a backend created by ie_create_backend. Null is a safe no-op.
IE_API void ie_destroy_backend(ie::Backend* backend);
// Returns the frozen API version string ("0.1"), valid for the process
// lifetime.
IE_API const char* ie_api_version();
}
