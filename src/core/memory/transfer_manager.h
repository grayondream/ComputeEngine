#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <utility>

#include "core/base/error.h"
#include "core/base/types.h"
#include "core/memory/allocator.h"
#include "core/runtime/stream.h"

namespace ie {

// Cross-device transfer surface (docs/readme.md §7.5). Routes may be
// registered per (source, destination) device pair; unregistered same-device
// transfers fall back to memcpy, cross-device transfers are Unsupported.
class TransferManager {
public:
    using TransferHandle = std::uint64_t;
    using TransferFn = std::function<Status(BufferHandle, Device, BufferHandle,
                                            Device, std::size_t)>;

    void register_route(Device from, Device to, TransferFn fn) {
        routes_[std::make_pair(from, to)] = std::move(fn);
    }

    Status transfer(BufferHandle dst, Device dst_dev, BufferHandle src,
                    Device src_dev, std::size_t bytes) {
        // Reject null buffers before the dst == src shortcut: otherwise
        // transfer(nullptr, dev, nullptr, dev, n > 0) would report success.
        if (bytes > 0 && (dst == nullptr || src == nullptr)) {
            return Error::invalid_argument("null buffer for transfer");
        }
        // Zero-byte and aliasing same-device transfers are already complete.
        // The same-device guard preserves the frozen contract that a
        // cross-device null/null zero-byte request is still Unsupported.
        if (src_dev == dst_dev && (bytes == 0 || dst == src)) {
            return {};
        }
        auto it = routes_.find(std::make_pair(src_dev, dst_dev));
        if (it != routes_.end()) {
            if (!it->second) {
                return Error::unsupported("empty transfer route");
            }
            return it->second(dst, dst_dev, src, src_dev, bytes);
        }
        if (src_dev != dst_dev) {
            return Error::unsupported(
                "cross-device transfer not implemented in P0.5");
        }
        // Same device, non-zero bytes, distinct non-null buffers.
        std::memcpy(dst, src, bytes);
        return {};
    }

    Status prefetch_async(BufferHandle, const void*, Stream, std::size_t) {
        return Error::unsupported("prefetch_async not implemented in P0.5");
    }

    Status wait(TransferHandle) {
        return Error::unsupported("transfer wait not implemented in P0.5");
    }

    // P0 compatibility entry point. Same-device copy is an explicit no-op
    // (the handle is not copied or duplicated). Cross-device callers must use
    // transfer() with explicit src/dst handles.
    Status copy(Device src, Device dst, BufferHandle src_handle,
                std::size_t bytes) {
        if (src != dst) {
            return Error::unsupported("cross-device copy requires transfer()");
        }
        if (bytes == 0) {
            return {};
        }
        if (src_handle == nullptr) {
            return Error::invalid_argument("null buffer for same-device copy");
        }
        return {};
    }

private:
    std::map<std::pair<Device, Device>, TransferFn> routes_;
};

}  // namespace ie
