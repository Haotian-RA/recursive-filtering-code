#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "recursive_filter.h"
#include <numeric>

#ifdef DOCTEST_LIBRARY_INCLUDED

using T = float;

// second order filter coefficients and initial conditions
T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

// testing for SSE
TEST_CASE("option accuracy test for M=4:") {
    using V = Vec4f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // Option 2
    IirCoreOrderTwo<V> I_op2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op2.option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // Option 3
    IirCoreOrderTwo<V> I_op3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op3.option3(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));

};

// testing for AVX2
TEST_CASE("option accuracy test for M=8:") {
    using V = Vec8f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // Option 2
    IirCoreOrderTwo<V> I_op2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op2.option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // Option 3
    IirCoreOrderTwo<V> I_op3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op3.option3(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));

};

// testing for AVX512
TEST_CASE("option accuracy test for M=16:") {
    using V = Vec16f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben.benchmark(data[n]);

    // option 1
    IirCoreOrderTwo<V> I_op1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op1.option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // Option 2
    IirCoreOrderTwo<V> I_op2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op2.option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // Option 3
    IirCoreOrderTwo<V> I_op3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op3.option3(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));

};

TEST_SUITE_END();

#endif // doctest
