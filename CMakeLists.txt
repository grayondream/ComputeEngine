cmake_minimum_required(VERSION 3.28)
project(compute_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ie_core STATIC
    src/core/base/shape.cpp
    src/core/base/types.cpp
    src/core/ir/graph.cpp
)
target_include_directories(ie_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ie_core PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(ie_core PUBLIC Threads::Threads)

enable_testing()
find_package(GTest REQUIRED)
add_subdirectory(tests)
