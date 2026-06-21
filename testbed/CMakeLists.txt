add_executable(testbed main.cpp)

target_link_libraries(testbed PRIVATE compute_engine)

target_include_directories(testbed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_definitions(testbed PRIVATE
    CE_TESTBED=1
)
