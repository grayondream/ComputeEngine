#pragma once

#include <cstddef>
#include <string>

#include "core/hardware/capability_set.h"

namespace ie {

// Generic hardware capability profile (docs/readme.md §2.1). Identity fields
// are for logging only; decisions use caps and the measured parameters.
struct HardwareProfile {
    // Identity (logging/debug only)
    std::string vendor;
    std::string device_name;
    std::string driver_version;
    std::string fingerprint;

    CapabilitySet caps;

    struct ComputeUnits {
        int num_cores{0};
        int threads_per_core{0};
        int max_threads_per_core{0};
        std::size_t shared_memory_per_core{0};
        std::size_t l2_cache_size{0};
        std::size_t l3_cache_size{0};
        int num_memory_channels{0};
    } cu;

    struct MatrixUnits {
        bool has_vector_unit{false};
        bool has_matrix_unit{false};
        int matrix_unit_generation{0};

        // CPU vector ISA
        bool has_sse2{false}, has_avx{false}, has_avx2{false};
        bool has_avx512f{false}, has_avx512bw{false}, has_avx512vl{false};
        bool has_avx512_vnni{false};
        bool has_avx512_bf16{false};
        bool has_avx512_fp16{false};
        bool has_amx_bf16{false}, has_amx_int8{false}, has_amx_fp16{false};
        bool has_neon{false}, has_sve{false}, has_sve2{false}, has_sme{false},
            has_sme2{false};

        // GPU matrix ISA
        bool has_tensor_core{false};
        bool supports_fp16_mma{false}, supports_bf16_mma{false},
            supports_tf32_mma{false};
        bool supports_int8_mma{false}, supports_int4_mma{false};
        bool supports_fp8_mma{false}, supports_fp4_mma{false};
        bool supports_sparsity{false};

        // Performance parameters
        float fp32_tflops{0.0f}, fp16_tflops{0.0f}, bf16_tflops{0.0f};
        float fp8_tflops{0.0f}, int8_tops{0.0f}, fp4_tflops{0.0f};
        float tensor_to_vector_ratio{0.0f};  // rho
    } matrix;

    struct Memory {
        std::size_t vram_total{0}, vram_available{0};
        float bandwidth_gbps{0.0f};
        std::string type;
        int bus_width{0};
        int memory_levels{0};
        bool has_error_correction{false};
    } mem;

    struct DataMovement {
        bool supports_async_copy{false};
        bool supports_bulk_tensor_copy{false};
        bool supports_p2p{false};
        bool supports_unified_memory{false};
        int max_copy_engines{0};
    } dma;

    struct ExecutionFeatures {
        bool supports_cooperative_groups{false};
        bool supports_workgroup_cluster{false};
        bool supports_distributed_shared_mem{false};
        bool supports_warp_specialization{false};
        bool supports_async_pipeline{false};
        int max_blocks_per_core{0};
        int max_warps_per_core{0};
    } exec;

    struct Interconnect {
        int pcie_gen{0}, pcie_lanes{0};
        float pcie_bandwidth_gbps{0.0f};
        bool has_nvlink{false};
        int nvlink_generation{0};
        float nvlink_bandwidth_gbps{0.0f};
        int nvlink_link_count{0};
        bool has_nvswitch{false};
    } link;

    struct Deployment {
        bool is_mig_enabled{false};
        int mig_slice_count{0};
        bool is_vgpu{false};
        bool is_consumer_grade{false};
        int tdp_watts{0};
        std::string power_mode;
        bool is_numa{false};
        int numa_nodes{0};
    } deploy;
};

}  // namespace ie
