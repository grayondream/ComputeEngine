#include "api/c_api/ie_api.h"

#include <new>
#include <string_view>

#include "backends/cpu/cpu_backend.h"

IE_API ie::Backend* ie_create_backend(const char* kind, const char* config) {
    (void)config;
    if (kind == nullptr) {
        return nullptr;
    }
    if (std::string_view(kind) != "cpu") {
        return nullptr;
    }
    auto* backend = new (std::nothrow) ie::CpuBackend();
    if (backend == nullptr) {
        return nullptr;
    }
    if (!backend->init().has_value()) {
        delete backend;
        return nullptr;
    }
    return backend;
}

IE_API void ie_destroy_backend(ie::Backend* backend) {
    if (backend == nullptr) {
        return;
    }
    backend->shutdown();
    delete backend;
}

IE_API const char* ie_api_version() { return ie::kApiVersionString; }
