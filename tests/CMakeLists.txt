add_executable(ie_tests
    base/shape_test.cpp
    ir/graph_test.cpp
    memory/memory_test.cpp
    backend/backend_test.cpp
    ops/ops_test.cpp
    hardware/hardware_test.cpp
    scheduler/scheduler_test.cpp
    runtime/engine_e2e_test.cpp
    plugin/plugin_test.cpp
    interface/config_contract_test.cpp
    interface/compiler_tuning_contract_test.cpp
    interface/hybrid_runtime_contract_test.cpp
)
target_link_libraries(ie_tests PRIVATE ie_cpu GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ie_tests)
