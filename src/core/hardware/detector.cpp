#include "core/hardware/detector.h"

#include <iomanip>
#include <sstream>
#include <string_view>
#include <thread>

namespace ie {
namespace {

void mix(std::uint64_t& hash, std::string_view text) {
    for (char ch : text) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(ch));
        hash *= 1099511628211ull;
    }
}

void enable(HardwareProfile& profile, const char* capability, bool value,
            bool HardwareProfile::MatrixUnits::*field) {
    if (value) {
        profile.matrix.*field = true;
        profile.caps.set(capability, true);
    }
}

}  // namespace

Result<HardwareProfile> HardwareDetector::detect_cpu() {
    HardwareProfile profile;
    profile.vendor = "generic";
    profile.device_name = "generic-cpu";
    profile.driver_version = "n/a";

    const unsigned cores = std::thread::hardware_concurrency();
    profile.cu.num_cores = cores == 0 ? 1 : static_cast<int>(cores);

#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    const bool sse2 = __builtin_cpu_supports("sse2");
    const bool avx = __builtin_cpu_supports("avx");
    const bool avx2 = __builtin_cpu_supports("avx2");
    const bool avx512f = __builtin_cpu_supports("avx512f");
    const bool avx512bw = __builtin_cpu_supports("avx512bw");
    const bool avx512vl = __builtin_cpu_supports("avx512vl");
    const bool avx512vnni = __builtin_cpu_supports("avx512vnni");
    const bool avx512bf16 = __builtin_cpu_supports("avx512bf16");
    const bool avx512fp16 = __builtin_cpu_supports("avx512fp16");
    const bool amxbf16 = __builtin_cpu_supports("amx-bf16");
    const bool amxint8 = __builtin_cpu_supports("amx-int8");
    const bool amxfp16 = __builtin_cpu_supports("amx-fp16");

    enable(profile, "sse2", sse2, &HardwareProfile::MatrixUnits::has_sse2);
    enable(profile, "avx", avx, &HardwareProfile::MatrixUnits::has_avx);
    enable(profile, "avx2", avx2, &HardwareProfile::MatrixUnits::has_avx2);
    enable(profile, "avx512f", avx512f,
           &HardwareProfile::MatrixUnits::has_avx512f);
    enable(profile, "avx512bw", avx512bw,
           &HardwareProfile::MatrixUnits::has_avx512bw);
    enable(profile, "avx512vl", avx512vl,
           &HardwareProfile::MatrixUnits::has_avx512vl);
    enable(profile, "avx512vnni", avx512vnni,
           &HardwareProfile::MatrixUnits::has_avx512_vnni);
    enable(profile, "avx512bf16", avx512bf16,
           &HardwareProfile::MatrixUnits::has_avx512_bf16);
    enable(profile, "avx512fp16", avx512fp16,
           &HardwareProfile::MatrixUnits::has_avx512_fp16);
    enable(profile, "amx-bf16", amxbf16,
           &HardwareProfile::MatrixUnits::has_amx_bf16);
    enable(profile, "amx-int8", amxint8,
           &HardwareProfile::MatrixUnits::has_amx_int8);
    enable(profile, "amx-fp16", amxfp16,
           &HardwareProfile::MatrixUnits::has_amx_fp16);
#endif

    profile.matrix.has_vector_unit = profile.matrix.has_sse2 ||
                                     profile.matrix.has_neon;
    profile.matrix.has_matrix_unit = profile.matrix.has_avx512_vnni ||
                                     profile.matrix.has_amx_int8 ||
                                     profile.matrix.has_tensor_core;
    profile.mem.type = "host";
    profile.mem.memory_levels = 1;

    profile.fingerprint = make_fingerprint(profile);
    return profile;
}

Result<HardwareProfile> HardwareDetector::detect_cuda(int /*device_id*/) {
    return Error::unsupported("gpu detection not implemented in P0");
}

Result<HardwareProfile> HardwareDetector::detect_vulkan(int /*device_id*/) {
    return Error::unsupported("gpu detection not implemented in P0");
}

std::string HardwareDetector::make_fingerprint(
    const HardwareProfile& profile) {
    std::uint64_t hash = 1469598103934665603ull;
    mix(hash, profile.vendor);
    mix(hash, profile.device_name);
    mix(hash, std::to_string(profile.cu.num_cores));
    for (const std::string& cap : profile.caps.list()) {
        mix(hash, cap);
    }
    std::ostringstream out;
    out << "cpu-" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

}  // namespace ie
