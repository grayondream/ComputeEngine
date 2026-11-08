#include "core/hardware/detector.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif

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

#if defined(__linux__)

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool read_text_file(const std::string& path, std::string& text) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    text = buffer.str();
    return true;
}

std::size_t parse_size_bytes(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size() &&
           (text[index] == ' ' || text[index] == '\t')) {
        ++index;
    }
    const std::size_t first_digit = index;
    std::size_t value = 0;
    while (index < text.size() && text[index] >= '0' && text[index] <= '9') {
        value = value * 10 + static_cast<std::size_t>(text[index] - '0');
        ++index;
    }
    if (index == first_digit) {
        return 0;
    }
    if (index < text.size()) {
        switch (text[index]) {
            case 'K':
            case 'k':
                value <<= 10;
                break;
            case 'M':
            case 'm':
                value <<= 20;
                break;
            case 'G':
            case 'g':
                value <<= 30;
                break;
            default:
                break;
        }
    }
    return value;
}

// Parses Linux cpu-list syntax such as "0", "0-3", or "0-1,4" and returns the
// number of listed CPUs/nodes. Returns 0 when the input has no numbers.
int count_cpu_list(std::string_view text) {
    int count = 0;
    std::size_t index = 0;
    while (index < text.size()) {
        while (index < text.size() &&
               (text[index] < '0' || text[index] > '9')) {
            ++index;
        }
        if (index >= text.size()) {
            break;
        }
        int first = 0;
        while (index < text.size() && text[index] >= '0' &&
               text[index] <= '9') {
            first = first * 10 + (text[index] - '0');
            ++index;
        }
        int last = first;
        if (index < text.size() && text[index] == '-') {
            ++index;
            last = 0;
            while (index < text.size() && text[index] >= '0' &&
                   text[index] <= '9') {
                last = last * 10 + (text[index] - '0');
                ++index;
            }
        }
        if (last >= first) {
            count += last - first + 1;
        }
    }
    return count;
}

std::size_t read_meminfo_bytes(std::string_view key) {
    std::ifstream input("/proc/meminfo");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(key, 0) != 0) {
            continue;
        }
        return parse_size_bytes(std::string_view(line).substr(key.size())) *
               1024;
    }
    return 0;
}

std::size_t read_cache_size(int level) {
    std::size_t size = 0;
    for (int index = 0; index < 32; ++index) {
        const std::string base = "/sys/devices/system/cpu/cpu0/cache/index" +
                                 std::to_string(index);
        std::string level_text;
        if (!read_text_file(base + "/level", level_text)) {
            continue;
        }
        if (static_cast<int>(parse_size_bytes(level_text)) != level) {
            continue;
        }
        std::string size_text;
        if (!read_text_file(base + "/size", size_text)) {
            continue;
        }
        size = std::max(size, parse_size_bytes(size_text));
    }
    return size;
}

int read_numa_nodes() {
    std::string text;
    if (!read_text_file("/sys/devices/system/node/online", text)) {
        return 1;
    }
    const int nodes = count_cpu_list(trim(text));
    return nodes > 0 ? nodes : 1;
}

struct ProcCpuInfo {
    std::string vendor;
    std::string device_name;
    int logical_cpus{0};
    int physical_cores{0};
};

ProcCpuInfo read_proc_cpuinfo() {
    ProcCpuInfo info;
    std::ifstream input("/proc/cpuinfo");
    if (!input) {
        return info;
    }

    std::set<std::pair<std::string, std::string>> cores;
    std::string physical_id;
    std::string core_id;
    const auto flush_core = [&] {
        if (!physical_id.empty() && !core_id.empty()) {
            cores.emplace(physical_id, core_id);
        }
        physical_id.clear();
        core_id.clear();
    };

    std::string line;
    while (std::getline(input, line)) {
        if (trim(line).empty()) {
            flush_core();
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));
        if (key == "processor") {
            ++info.logical_cpus;
        } else if (key == "vendor_id") {
            if (info.vendor.empty()) {
                info.vendor = value;
            }
        } else if (key == "model name") {
            if (info.device_name.empty()) {
                info.device_name = value;
            }
        } else if (key == "physical id") {
            physical_id = value;
        } else if (key == "core id") {
            core_id = value;
        }
    }
    flush_core();
    info.physical_cores = static_cast<int>(cores.size());
    return info;
}

#endif  // defined(__linux__)

CpuFeatures detect_cpu_features() {
    CpuFeatures features;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    const bool fma = __builtin_cpu_supports("fma");
    features.sse2 = __builtin_cpu_supports("sse2");
    features.avx = __builtin_cpu_supports("avx");
    features.avx2 = __builtin_cpu_supports("avx2");
    features.fma = fma;
    features.avx512f = __builtin_cpu_supports("avx512f");
    features.avx512bw = __builtin_cpu_supports("avx512bw");
    features.avx512vl = __builtin_cpu_supports("avx512vl");
    features.avx512_vnni = __builtin_cpu_supports("avx512vnni");
    features.avx512_bf16 = __builtin_cpu_supports("avx512bf16");
    features.avx512_fp16 = __builtin_cpu_supports("avx512fp16");

    // AMX bits are not uniformly exposed by __builtin_cpu_supports across
    // compiler versions, so query CPUID leaf 7 directly.
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    bool amx_tile = false;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
        amx_tile = (edx & (1u << 24)) != 0;
        features.amx_bf16 = amx_tile && (edx & (1u << 22)) != 0;
        features.amx_int8 = amx_tile && (edx & (1u << 25)) != 0;
    }
    if (amx_tile && __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx) != 0) {
        features.amx_fp16 = (eax & (1u << 21)) != 0;
    }
#elif defined(__linux__) && defined(__aarch64__)
    // On aarch64 NEON is HWCAP_ASIMD (bit 1), not the arm32 HWCAP_NEON bit.
#ifdef HWCAP_ASIMD
    constexpr unsigned long kHwcapAsimd = HWCAP_ASIMD;
#else
    constexpr unsigned long kHwcapAsimd = 1ul << 1;
#endif
#ifdef HWCAP_SVE
    constexpr unsigned long kHwcapSve = HWCAP_SVE;
#else
    constexpr unsigned long kHwcapSve = 1ul << 22;
#endif
#ifdef HWCAP2_SVE2
    constexpr unsigned long kHwcap2Sve2 = HWCAP2_SVE2;
#else
    constexpr unsigned long kHwcap2Sve2 = 1ul << 1;
#endif
#ifdef HWCAP2_SME
    constexpr unsigned long kHwcap2Sme = HWCAP2_SME;
#else
    constexpr unsigned long kHwcap2Sme = 1ul << 23;
#endif
#ifdef HWCAP2_SME2
    constexpr unsigned long kHwcap2Sme2 = HWCAP2_SME2;
#else
    constexpr unsigned long kHwcap2Sme2 = 1ul << 37;
#endif

    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    features.neon = (hwcap & kHwcapAsimd) != 0;
    features.sve = (hwcap & kHwcapSve) != 0;
    features.sve2 = (hwcap2 & kHwcap2Sve2) != 0;
    features.sme = (hwcap2 & kHwcap2Sme) != 0;
    features.sme2 = (hwcap2 & kHwcap2Sme2) != 0;
#elif defined(__linux__) && defined(__arm__)
#ifdef HWCAP_NEON
    constexpr unsigned long kHwcapNeon = HWCAP_NEON;
#else
    constexpr unsigned long kHwcapNeon = 1ul << 12;
#endif
    const unsigned long hwcap = getauxval(AT_HWCAP);
    features.neon = (hwcap & kHwcapNeon) != 0;
#endif
    return features;
}

CpuTopology detect_cpu_topology() {
    CpuTopology topology;
    topology.vendor = "generic";
    topology.device_name = "generic";
    topology.num_memory_channels = 1;
    topology.numa_nodes = 1;

    const unsigned concurrent = std::thread::hardware_concurrency();
    int logical_cpus = concurrent == 0 ? 1 : static_cast<int>(concurrent);

#if defined(__linux__)
    const ProcCpuInfo info = read_proc_cpuinfo();
    if (!info.vendor.empty()) {
        topology.vendor = info.vendor;
    }
    if (!info.device_name.empty()) {
        topology.device_name = info.device_name;
    }
    if (info.logical_cpus > 0) {
        logical_cpus = info.logical_cpus;
    }
    topology.num_cores = info.physical_cores;
    topology.total_memory = read_meminfo_bytes("MemTotal:");
    topology.available_memory = read_meminfo_bytes("MemAvailable:");
    topology.l2_cache_size = read_cache_size(2);
    topology.l3_cache_size = read_cache_size(3);
    topology.numa_nodes = read_numa_nodes();
#endif

    if (topology.num_cores <= 0) {
        topology.num_cores = logical_cpus;
    }
    topology.threads_per_core =
        std::max(1, logical_cpus / topology.num_cores);
    topology.max_threads_per_core = topology.threads_per_core;
    if (topology.numa_nodes <= 0) {
        topology.numa_nodes = 1;
    }
    return topology;
}

}  // namespace

Result<HardwareProfile> HardwareDetector::detect_cpu() {
    return build_cpu_profile(detect_cpu_features(), detect_cpu_topology());
}

HardwareProfile HardwareDetector::build_cpu_profile(
    const CpuFeatures& features, const CpuTopology& topology) {
    HardwareProfile profile;
    profile.vendor = topology.vendor.empty() ? "generic" : topology.vendor;
    profile.device_name =
        topology.device_name.empty() ? "generic" : topology.device_name;
    profile.driver_version = "n/a";

    profile.cu.num_cores = topology.num_cores;
    profile.cu.threads_per_core = topology.threads_per_core;
    profile.cu.max_threads_per_core = topology.max_threads_per_core;
    profile.cu.l2_cache_size = topology.l2_cache_size;
    profile.cu.l3_cache_size = topology.l3_cache_size;
    profile.cu.num_memory_channels = topology.num_memory_channels;

    profile.mem.vram_total = topology.total_memory;
    profile.mem.vram_available = topology.available_memory;
    profile.mem.type = "host";
    profile.mem.memory_levels = 1;

    profile.deploy.is_numa = topology.numa_nodes > 1;
    profile.deploy.numa_nodes = topology.numa_nodes;

    enable(profile, "sse2", features.sse2,
           &HardwareProfile::MatrixUnits::has_sse2);
    enable(profile, "avx", features.avx,
           &HardwareProfile::MatrixUnits::has_avx);
    enable(profile, "avx2", features.avx2,
           &HardwareProfile::MatrixUnits::has_avx2);
    if (features.fma) {
        profile.caps.set("fma", true);
    }
    enable(profile, "avx512f", features.avx512f,
           &HardwareProfile::MatrixUnits::has_avx512f);
    enable(profile, "avx512bw", features.avx512bw,
           &HardwareProfile::MatrixUnits::has_avx512bw);
    enable(profile, "avx512vl", features.avx512vl,
           &HardwareProfile::MatrixUnits::has_avx512vl);
    enable(profile, "avx512vnni", features.avx512_vnni,
           &HardwareProfile::MatrixUnits::has_avx512_vnni);
    enable(profile, "avx512bf16", features.avx512_bf16,
           &HardwareProfile::MatrixUnits::has_avx512_bf16);
    enable(profile, "avx512fp16", features.avx512_fp16,
           &HardwareProfile::MatrixUnits::has_avx512_fp16);
    enable(profile, "amx-bf16", features.amx_bf16,
           &HardwareProfile::MatrixUnits::has_amx_bf16);
    enable(profile, "amx-int8", features.amx_int8,
           &HardwareProfile::MatrixUnits::has_amx_int8);
    enable(profile, "amx-fp16", features.amx_fp16,
           &HardwareProfile::MatrixUnits::has_amx_fp16);
    enable(profile, "neon", features.neon,
           &HardwareProfile::MatrixUnits::has_neon);
    enable(profile, "sve", features.sve,
           &HardwareProfile::MatrixUnits::has_sve);
    enable(profile, "sve2", features.sve2,
           &HardwareProfile::MatrixUnits::has_sve2);
    enable(profile, "sme", features.sme,
           &HardwareProfile::MatrixUnits::has_sme);
    enable(profile, "sme2", features.sme2,
           &HardwareProfile::MatrixUnits::has_sme2);

    const bool has_amx = features.amx_bf16 || features.amx_int8 ||
                         features.amx_fp16;
    profile.matrix.has_vector_unit = features.sse2 || features.neon;
    profile.matrix.has_matrix_unit = features.avx512_vnni ||
                                     features.amx_int8 ||
                                     features.amx_bf16;
    if (has_amx) {
        profile.matrix.matrix_unit_generation = 2;
    } else if (features.avx512_vnni) {
        profile.matrix.matrix_unit_generation = 1;
    }

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
    mix(hash, std::to_string(profile.cu.l2_cache_size));
    mix(hash, std::to_string(profile.cu.l3_cache_size));
    for (const std::string& cap : profile.caps.list()) {
        mix(hash, cap);
    }
    std::ostringstream out;
    out << "cpu-" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

}  // namespace ie
