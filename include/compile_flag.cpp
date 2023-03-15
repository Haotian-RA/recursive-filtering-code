#include "instrset_detect.h"
#include <iostream>

// clang++ -I/usr/local/include -I$VCL_PATH -std=c++20 -o compile_flag compile_flag.cpp

int main(){

    std::cout<<instrset_detect()<<std::endl; // 8. accepts avx2.
    std::cout<<hasFMA3()<<std::endl; // 1. accepts mfma.

    return 0;

}