cmake_minimum_required(VERSION 3.22)
project(ComputeEngine VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(cmake/CompilerOptions.cmake)
include(cmake/FindBackends.cmake)

# ─── SPDLOG (header-only) ───
include(FetchContent)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
endif()
set(SPDLOG_INCLUDE_DIR ${spdlog_SOURCE_DIR}/include)

# ─── Build Options ───
option(CE_BUILD_CUDA     "Build CUDA backend"      ON)
option(CE_BUILD_OPENCL   "Build OpenCL backend"     ON)
option(CE_BUILD_VULKAN   "Build Vulkan backend"     ON)
option(CE_BUILD_AVX      "Enable AVX/AVX2/FMA"      ON)
option(CE_BUILD_NEON     "Enable ARM NEON"           OFF)
option(CE_BUILD_TESTBED  "Build testbed"            ON)
option(CE_BUILD_SHARED   "Build shared library"     ON)
option(CE_ENABLE_PROFILING "Enable Profiler"        ON)
option(CE_ENABLE_LOGGING  "Enable Logger"           ON)
option(CE_DEBUG_TRACE    "Enable resource tracing"  OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Source Files ───
set(CE_CORE_SOURCES
    src/core/tensor.cpp
    src/core/half.cpp
    src/core/status.cpp
)

set(CE_PLATFORM_SOURCES
    src/platform/thread_pool.cpp
    src/platform/logger.cpp
    src/platform/profiler.cpp
    src/platform/timer.cpp
)

set(CE_BACKEND_COMMON_SOURCES
    src/backend/backend_registry.cpp
)

set(CE_MEMORY_SOURCES
    src/memory/allocator.cpp
    src/memory/memory_pool.cpp
    src/memory/unified_memory.cpp
    src/memory/resource_tracer.cpp
)

set(CE_OP_SOURCES
    src/op/op_registry.cpp
    src/op/ref_ops.cpp
)

set(CE_GRAPH_SOURCES
    src/graph/graph.cpp
    src/graph/graph_optimizer.cpp
    src/graph/graph_executor.cpp
    src/graph/onnx_parser.cpp
    src/graph/model.cpp
)

set(CE_RUNTIME_SOURCES
    src/runtime/session.cpp
    src/runtime/pipeline.cpp
    src/runtime/model_registry.cpp
)

set(CE_LLM_SOURCES
    src/llm/llm_engine.cpp
    src/llm/attention.cpp
    src/llm/kv_cache.cpp
    src/llm/paged_attention.cpp
    src/llm/scheduler.cpp
    src/llm/sampler.cpp
    src/llm/tokenizer.cpp
    src/llm/model_config.cpp
)

set(CE_CPU_BACKEND_SOURCES
    src/backend/cpu/cpu_backend.cpp
    src/backend/cpu/dispatch.cpp
    src/backend/cpu/kernels/matmul/matmul_scalar.cpp
    src/backend/cpu/kernels/matmul/matmul_avx.cpp
    src/backend/cpu/kernels/matmul/matmul_avx512.cpp
    src/backend/cpu/kernels/matmul/matmul_neon.cpp
    src/backend/cpu/kernels/conv/conv_scalar.cpp
    src/backend/cpu/kernels/norm/norm_scalar.cpp
    src/backend/cpu/kernels/norm/norm_avx.cpp
    src/backend/cpu/kernels/activation/activation_scalar.cpp
    src/backend/cpu/kernels/activation/activation_avx.cpp
)

set(CE_CUDA_BACKEND_SOURCES
    src/backend/cuda/cuda_backend.cpp
    src/backend/cuda/cuda_context.cpp
    src/backend/cuda/cuda_memory.cpp
)

set(CE_OPENCL_BACKEND_SOURCES
    src/backend/opencl/opencl_backend.cpp
    src/backend/opencl/cl_context.cpp
    src/backend/opencl/cl_kernel_compiler.cpp
)

set(CE_VULKAN_BACKEND_SOURCES
    src/backend/vulkan/vulkan_backend.cpp
    src/backend/vulkan/vk_context.cpp
    src/backend/vulkan/vk_compute_pipeline.cpp
)

# ─── Collect all library sources ───
set(CE_ALL_SOURCES
    ${CE_CORE_SOURCES}
    ${CE_PLATFORM_SOURCES}
    ${CE_BACKEND_COMMON_SOURCES}
    ${CE_MEMORY_SOURCES}
    ${CE_OP_SOURCES}
    ${CE_GRAPH_SOURCES}
    ${CE_RUNTIME_SOURCES}
    ${CE_LLM_SOURCES}
)

# ─── CPU Backend (always available) ───
list(APPEND CE_ALL_SOURCES ${CE_CPU_BACKEND_SOURCES})

# ─── Optional backends ───
if(CE_BUILD_CUDA)
    list(APPEND CE_ALL_SOURCES ${CE_CUDA_BACKEND_SOURCES})
endif()

if(CE_BUILD_OPENCL)
    list(APPEND CE_ALL_SOURCES ${CE_OPENCL_BACKEND_SOURCES})
endif()

if(CE_BUILD_VULKAN)
    list(APPEND CE_ALL_SOURCES ${CE_VULKAN_BACKEND_SOURCES})
endif()

# ─── Create library ───
if(CE_BUILD_SHARED)
    add_library(compute_engine SHARED ${CE_ALL_SOURCES})
else()
    add_library(compute_engine STATIC ${CE_ALL_SOURCES})
endif()

target_include_directories(compute_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${SPDLOG_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_compile_definitions(compute_engine PRIVATE
    CE_EXPORT_SYMBOLS
)

if(CE_ENABLE_PROFILING)
    target_compile_definitions(compute_engine PUBLIC CE_ENABLE_PROFILING=1)
endif()

if(CE_ENABLE_LOGGING)
    target_compile_definitions(compute_engine PUBLIC CE_ENABLE_LOGGING=1)
endif()

if(CE_DEBUG_TRACE)
    target_compile_definitions(compute_engine PUBLIC CE_DEBUG_TRACE=1)
endif()

# ─── Backend-specific linking ───
find_and_link_backends(compute_engine)

# ─── Threading ───
find_package(Threads REQUIRED)
target_link_libraries(compute_engine PUBLIC Threads::Threads)

# ─── Testbed ───
if(CE_BUILD_TESTBED)
    add_subdirectory(testbed)
endif()

# ─── Install ───
install(TARGETS compute_engine
    EXPORT ComputeEngineTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/compute_engine
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ComputeEngineTargets
    FILE ComputeEngineTargets.cmake
    NAMESPACE ComputeEngine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ComputeEngine
)
