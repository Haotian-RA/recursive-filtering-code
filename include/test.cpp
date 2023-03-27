/*    
    clang++ -I/usr/local/include -mavx2 -mfma -march=native -fno-trapping-math -fno-math-errno -I$VCL_PATH -std=c++20 -O3 -o test test.cpp
    clang++ -I/usr/local/include -mavx2 -mfma -march=native -I$VCL_PATH -std=c++20 -O3 -o test test.cpp

*/

#include "shift_reg.h"
#include "filter.h"
#include <chrono>
#include <iostream>


// test the time for filtering 1M samples.

using V = Vec8f;
using T = decltype(std::declval<V>().extract(0));
constexpr int M = V::size();
constexpr int L = 12;


int main(){


    T inits[L/2][4] = {0.2,0.3,0.4,0.5
                      ,0.5,0.7,0.9,3
                      ,-2,2.1,3,9
                      ,2.2,2.3,0.5,0.3
                      ,-2,-3,5,7
                      ,2,3,1,8
                      };

    T coefs[L/2][5] = {-0.5,0.25,-0.75,0.6
                      ,0.5,0.7,0.9,0.1
                      ,-0.2,0.2,0.3,0.9
                      ,-0.4,0.5,0.5,0.1
                      ,-0.25,-0.3,0.15,0.7
                      ,0.12,0.23,0.31,0.8
                      };

    static const int vector_size = 1024000;
    std::array<T,vector_size> in = {0}, out;
    in[0] = 1;

    Filter F(coefs,inits);

    auto start = std::chrono::high_resolution_clock::now();

    // auto r = F.Cascaded_option_1(in.begin(),in.end(),out.begin());
    // auto r = F.Cascaded_option_2(in.begin(),in.end(),out.begin());
    auto r = F.Cascaded_option_3(in.begin(),in.end(),out.begin());

    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

    return 0;

}





