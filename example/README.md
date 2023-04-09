### recommand compiler flags: 
clang++ -I/usr/local/include -mavx2 -mfma -march=native -fno-trapping-math -fno-math-errno -I$VCL_PATH -std=c++20 -O3 -o filter filter.cpp
