
cmake_minimum_required(VERSION 3.10)

project(TEST)

# Compiler Options
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -I/usr/local/include -march=native -mavx2 -mfma -O3")

include_directories($(Test_SOURCE_DIR)/src/vcl)

add_executable(TEST test/option1.cpp)

