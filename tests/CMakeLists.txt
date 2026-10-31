add_executable(ie_tests
    base/shape_test.cpp
)
target_link_libraries(ie_tests PRIVATE ie_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ie_tests)
