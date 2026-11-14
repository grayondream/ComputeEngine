add_executable(ie_tests
    base/shape_test.cpp
    ir/graph_test.cpp
    ir/attr_test.cpp
    memory/memory_test.cpp
    memory/cpu_allocator_test.cpp
    backend/backend_test.cpp
    ops/ops_test.cpp
    ops/activation_test.cpp
    ops/simd_kernels_test.cpp
    ops/traditional_ops_test.cpp
    hardware/hardware_test.cpp
    hardware/cpu_kernel_catalog_test.cpp
    scheduler/scheduler_test.cpp
    runtime/engine_e2e_test.cpp
    runtime/mlp_e2e_test.cpp
    runtime/thread_pool_test.cpp
    performance/matmul_benchmark_test.cpp
    plugin/plugin_test.cpp
    models/onnx_loader_test.cpp
    interface/config_contract_test.cpp
    interface/compiler_tuning_contract_test.cpp
    interface/hybrid_runtime_contract_test.cpp
    interface/rules_contract_test.cpp
    interface/ops_families_contract_test.cpp
    interface/llm_contract_test.cpp
    interface/training_contract_test.cpp
    interface/validation_contract_test.cpp
    interface/converter_contract_test.cpp
    interface/api_support_contract_test.cpp
    interface/inventory_test.cpp
)
target_link_libraries(ie_tests PRIVATE ie_cpu GTest::gtest_main)
target_include_directories(ie_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ie_tests PRIVATE
    IE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
include(GoogleTest)
gtest_discover_tests(ie_tests)
