add_executable(testbed main.cpp)
target_link_libraries(testbed PRIVATE compute_engine)
target_include_directories(testbed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(testbed PRIVATE CE_TESTBED=1)

add_executable(test_unit test_unit.cpp)
target_link_libraries(test_unit PRIVATE compute_engine)
target_include_directories(test_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(test_unit PRIVATE CE_TESTBED=1)
