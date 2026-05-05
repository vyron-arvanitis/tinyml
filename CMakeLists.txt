cmake_minimum_required(VERSION 3.20)

# Project metadata + language selection (enables CMake's CXX toolchain logic)
project(tinyml VERSION 0.1.0 LANGUAGES CXX)

# -----------------------------
# C++ language standard settings
# -----------------------------
set(CMAKE_CXX_STANDARD 20)                # compile as C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)        # error out if compiler can't do C++20
set(CMAKE_CXX_EXTENSIONS OFF)              # prefer -std=c++20 over gnu++20

# -----------------------------
# Build target: main executable
# -----------------------------
add_executable(tinyml
        src/main.cpp
        src/matrix.cpp
        src/random_gen.cpp
        src/loss.cpp
        src/linear_regression.cpp
)

# -----------------------------
# Header search path
# -----------------------------
# Lets you do: #include "matrix.h" from files in src/
target_include_directories(tinyml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -----------------------------
# Compiler warnings (good practice)
# -----------------------------
# These flags are GCC/Clang-oriented. With MSVC you'd typically add /W4 instead.
target_compile_options(tinyml PRIVATE -Wall -Wextra -Wpedantic)


# -----------------------------
# OpenMP
# -----------------------------
if (APPLE)
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif ()

find_package(OpenMP REQUIRED)

target_link_libraries(tinyml PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------
# Tests
# -----------------------------
find_package(GTest REQUIRED)

enable_testing()

add_executable(tinyml_tests
        tests/tensor.cpp
)

target_include_directories(tinyml_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tinyml_tests PRIVATE
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tinyml_tests)

