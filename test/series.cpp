#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "recursive_filter.h"
#include <numeric>

#ifdef DOCTEST_LIBRARY_INCLUDED

using T = float;

// second order filter coefficients and initial conditions
T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

// testing for SSE
TEST_CASE("series accuracy test for M=4:") {
    using V = Vec4f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y, x_T, y_T;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben3.benchmark(I_ben2.benchmark(I_ben1.benchmark(data[n])));

    // series of option 1
    IirCoreOrderTwo<V> I_op1_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op1(I_op1_1, I_op1_2, I_op1_3);
    for (auto n=0; n<M; n++) y[n] = S_op1.series_option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // series of option 2
    IirCoreOrderTwo<V> I_op2_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op2(I_op2_1, I_op2_2, I_op2_3);
    y = S_op2.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // series of option 3
    IirCoreOrderTwo<V> I_op3_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op3(I_op3_1, I_op3_2, I_op3_3);
    x_T = _permuteV(x);
    y_T = S_op3.series_option3(x_T);
    y = _permuteV(y_T);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]);
    
    // series from make
    // define array of coefficients and initial conditions
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};
    
    auto S_make = series_from_coeffs<T,V>(coefs, inits);
    y = S_make.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]);
    
    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));
    
};

// testing for AVX2
TEST_CASE("series accuracy test for M=8:") {
    using V = Vec8f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y, x_T, y_T;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben3.benchmark(I_ben2.benchmark(I_ben1.benchmark(data[n])));

    // series of option 1
    IirCoreOrderTwo<V> I_op1_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op1(I_op1_1, I_op1_2, I_op1_3);
    for (auto n=0; n<M; n++) y[n] = S_op1.series_option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // series of option 2
    IirCoreOrderTwo<V> I_op2_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op2(I_op2_1, I_op2_2, I_op2_3);
    y = S_op2.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // series of option 3
    IirCoreOrderTwo<V> I_op3_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op3(I_op3_1, I_op3_2, I_op3_3);
    x_T = _permuteV(x);
    y_T = S_op3.series_option3(x_T);
    y = _permuteV(y_T);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // series from make
    // define array of coefficients and initial conditions
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};
    
    auto S_make = series_from_coeffs<T,V>(coefs, inits);
    y = S_make.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]);
    
    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));

};

// testing for AVX512
TEST_CASE("series accuracy test for M=16:") {
    using V = Vec16f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a matrix of data
    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<V,M> x, y, x_T, y_T;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    std::array<T, M*M> y_ben, y_op1, y_op2, y_op3, y_op4;

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<M*M; n++) y_ben[n] = I_ben3.benchmark(I_ben2.benchmark(I_ben1.benchmark(data[n])));

    // series of option 1
    IirCoreOrderTwo<V> I_op1_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op1_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op1(I_op1_1, I_op1_2, I_op1_3);
    for (auto n=0; n<M; n++) y[n] = S_op1.series_option1(x[n]);
    for (auto n=0; n<M; n++) y[n].store(&y_op1[n*M]); 

    // series of option 2
    IirCoreOrderTwo<V> I_op2_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op2_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op2(I_op2_1, I_op2_2, I_op2_3);
    y = S_op2.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op2[n*M]); 

    // series of option 3
    IirCoreOrderTwo<V> I_op3_1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_op3_3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S_op3(I_op3_1, I_op3_2, I_op3_3);
    x_T = _permuteV(x);
    y_T = S_op3.series_option3(x_T);
    y = _permuteV(y_T);
    for (auto n=0; n<M; n++) y[n].store(&y_op3[n*M]); 

    // series from make
    // define array of coefficients and initial conditions
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};
    
    auto S_make = series_from_coeffs<T,V>(coefs, inits);
    y = S_make.series_option2(x);
    for (auto n=0; n<M; n++) y[n].store(&y_op4[n*M]);
    
    // check accuracy of filter sample by sample
    for (auto n=0; n<M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));

};

TEST_SUITE_END();

#endif // doctest
