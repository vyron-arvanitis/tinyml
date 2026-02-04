cmake_minimum_required(VERSION 3.20)
project(tinyml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main executable
add_executable(tinyml
    src/main.cpp
    src/matrix.cpp
    src/random_gen.cpp
    src/loss.cpp
)

# Include path for headers
target_include_directories(tinyml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compiler warnings
target_compile_options(tinyml PRIVATE -Wall -Wextra -Wpedantic)
