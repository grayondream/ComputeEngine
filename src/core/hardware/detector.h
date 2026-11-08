#pragma once

#include <cstddef>
#include <string>

#include "core/base/error.h"
#include "core/hardware/capability_set.h"
#include "core/hardware/profile.h"

namespace ie {

// Raw CPU ISA capabilities detected by the platform probes. All flags default
// to false so a caller can describe a synthetic CPU in tests.
struct CpuFeatures {
    bool sse2{false};
    bool avx{false};
    bool avx2{false};
    bool fma{false};
    bool avx512f{false};
    bool avx512bw{false};
    bool avx512vl{false};
    bool avx512_vnni{false};
    bool avx512_bf16{false};
    bool avx512_fp16{false};
    bool amx_bf16{false};
    bool amx_int8{false};
    bool amx_fp16{false};
    bool neon{false};
    bool sve{false};
    bool sve2{false};
    bool sme{false};
    bool sme2{false};
};

// Raw CPU topology/memory information gathered from the platform. Values are
// converted into the generic HardwareProfile by build_cpu_profile().
struct CpuTopology {
    std::string vendor;
    std::string device_name;
    int num_cores{0};
    int threads_per_core{0};
    int max_threads_per_core{0};
    std::size_t l2_cache_size{0};
    std::size_t l3_cache_size{0};
    int num_memory_channels{0};
    std::size_t total_memory{0};
    std::size_t available_memory{0};
    int numa_nodes{0};
};

class HardwareDetector {
public:
    static Result<HardwareProfile> detect_cpu();
    static Result<HardwareProfile> detect_cuda(int device_id);
    static Result<HardwareProfile> detect_vulkan(int device_id);

    // Pure mapping from detected feature/topology data to the public profile.
    static HardwareProfile build_cpu_profile(const CpuFeatures& features,
                                             const CpuTopology& topology);

private:
    static std::string make_fingerprint(const HardwareProfile& profile);
};

}  // namespace ie
