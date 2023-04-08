#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "recursive_filter.h"
#include <numeric>

#ifdef DOCTEST_LIBRARY_INCLUDED

using T = float;

// second order filter coefficients and initial conditions
T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

// testing for SSE
TEST_CASE("cas_option accuracy test for M=4:") {
    using V = Vec4f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben2.benchmark(I_ben1.benchmark(data[n]));

    // option 1 + option 1
    IirCoreOrderTwo<V> I_op11_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op11_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op11_2.option1(I_op11_1.option1(x[n]));
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // option 2 + option 2
    IirCoreOrderTwo<V> I_op22_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op22_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op22_2.option2(I_op22_1.option2(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // option 3 + option 2
    IirCoreOrderTwo<V> I_op32_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op32_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op32_2.option2_tail(I_op32_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // option 3 + option 3
    IirCoreOrderTwo<V> I_op33_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op33_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op33_2.option3_tail(I_op33_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));

};

// testing for AVX2
TEST_CASE("cas_option accuracy test for M=8:") {
    using V = Vec8f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben2.benchmark(I_ben1.benchmark(data[n]));

    // option 1 + option 1
    IirCoreOrderTwo<V> I_op11_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op11_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op11_2.option1(I_op11_1.option1(x[n]));
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // option 2 + option 2
    IirCoreOrderTwo<V> I_op22_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op22_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op22_2.option2(I_op22_1.option2(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // option 3 + option 2
    IirCoreOrderTwo<V> I_op32_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op32_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op32_2.option2_tail(I_op32_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // option 3 + option 3
    IirCoreOrderTwo<V> I_op33_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op33_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op33_2.option3_tail(I_op33_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));

};

// testing for AVX512
TEST_CASE("cas_option accuracy test for M=16:") {
    using V = Vec16f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben2.benchmark(I_ben1.benchmark(data[n]));

    // option 1 + option 1
    IirCoreOrderTwo<V> I_op11_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op11_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M; n++) y[n] = I_op11_2.option1(I_op11_1.option1(x[n]));
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // option 2 + option 2
    IirCoreOrderTwo<V> I_op22_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op22_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op22_2.option2(I_op22_1.option2(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // option 3 + option 2
    IirCoreOrderTwo<V> I_op32_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op32_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op32_2.option2_tail(I_op32_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // option 3 + option 3
    IirCoreOrderTwo<V> I_op33_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op33_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    y = I_op33_2.option3_tail(I_op33_1.option3_head(x));
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]); 

    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));

};

TEST_SUITE_END();

#endif // doctest
