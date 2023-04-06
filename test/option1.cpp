#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../src/doctest.h"
#include "../include/second_order_cores.h"
#include <numeric>

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("option 1 accuracy test, M=4:") {
    // testing for SSE
    using V = Vec4f;

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // coefficients and initial conditions of a second order core
    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    // define a matrix of data as input to test both accuracy of shift register and second order core 
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
}

TEST_CASE("option 1 accuracy test, M=8:") {
    // testing for AVX2
    using V = Vec8f;

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // coefficients and initial conditions of a second order core
    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    // define a matrix of data as input to test both accuracy of shift register and second order core 
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
}

TEST_CASE("option 1 accuracy test, M=16:") {
    // testing for AVX512
    using V = Vec16f;

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // coefficients and initial conditions of a second order core
    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    // define a matrix of data as input to test both accuracy of shift register and second order core 
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
}

TEST_SUITE_END();

#endif // doctest
