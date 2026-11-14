cmake_minimum_required(VERSION 3.28)
project(compute_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ie_core STATIC
    src/core/base/shape.cpp
    src/core/base/types.cpp
    src/core/ir/graph.cpp
    src/core/backend/engine_config.cpp
    src/core/ops/op_registry.cpp
    src/core/hardware/detector.cpp
    src/core/hardware/optimization_planner.cpp
    src/core/scheduler/static_scheduler.cpp
    src/core/runtime/executor.cpp
    src/core/runtime/thread_pool.cpp
    src/ops/traditional/conv2d_op.cpp
    src/ops/traditional/pool_ops.cpp
    src/ops/traditional/batch_norm_op.cpp
    src/ops/traditional/gemm_op.cpp
    src/ops/traditional/layout_ops.cpp
    src/models/traditional/onnx/onnx_wire.cpp
    src/models/traditional/onnx/onnx_loader.cpp
    src/converter/format_loader/format_loader.cpp
    src/converter/op_mapping/onnx_mapping.cpp
)

add_library(ie_cpu STATIC
    src/backends/cpu/cpu_backend.cpp
    src/backends/cpu/kernels/cpu_kernel_catalog.cpp
    src/backends/cpu/simd/simd_kernels.cpp
    src/core/runtime/engine.cpp
    src/api/c_api/ie_c_api.cpp
)
target_link_libraries(ie_cpu PUBLIC ie_core)
target_compile_options(ie_cpu PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ie_demo src/demo/main.cpp)
target_link_libraries(ie_demo PRIVATE ie_cpu)
add_test(NAME demo_e2e COMMAND ie_demo)

add_executable(ie_bench src/benchmarks/ie_bench.cpp)
target_link_libraries(ie_bench PRIVATE ie_cpu)
target_compile_options(ie_bench PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ie_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ie_core PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(ie_core PUBLIC Threads::Threads)

enable_testing()
find_package(GTest REQUIRED)
add_subdirectory(tests)
