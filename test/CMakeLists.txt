
cmake_minimum_required(VERSION 3.10)

project(Test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

target_include_directories(Test PUBLIC src/vcl)



add_executable(Test option1.cpp)

