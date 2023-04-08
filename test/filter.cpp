#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "recursive_filter.h"
#include <numeric>

#ifdef DOCTEST_LIBRARY_INCLUDED

using T = float;

// second order filter coefficients and initial conditions
T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

// testing for SSE
TEST_CASE("filter accuracy test for M=8:") {
    using V = Vec8f;

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    // define a trunk of data (2 matrices)
    std::vector<T> data(2*M*M);
    std::iota(data.begin(), data.end(), 0); 

    std::array<T, 2*M*M> x, y_ben, y_op1, y_op2, y_op3, y_op4, y_op5;

    for (auto n=0; n<2*M*M; n++) x[n] = data[n];

    // benchmark (scalar)
    IirCoreOrderTwo<V> I_ben1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben2(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I_ben3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);
    for (auto n=0; n<2*M*M; n++) y_ben[n] = I_ben3.benchmark(I_ben2.benchmark(I_ben1.benchmark(data[n])));

    // define array of coefficients and initial conditions
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};

    // filter of scalar
    Filter F_op1(coefs,inits);
    F_op1.cascaded_option1(x.begin(),x.end(),y_op1.begin());

    // filter of option 1
    Filter F_op2(coefs,inits);
    F_op2.cascaded_option1(x.begin(),x.end(),y_op2.begin());

    // filter of option 2
    Filter F_op3(coefs,inits);
    F_op3.cascaded_option2(x.begin(),x.end(),y_op3.begin());

    // filter of option 3
    Filter F_op4(coefs,inits);
    F_op4.cascaded_option3(x.begin(),x.end(),y_op4.begin());

    // filter of operator
    Filter F_op5(coefs,inits);
    F_op5(x.begin(),x.end(),y_op5.begin());

    // check accuracy of filter sample by sample
    for (auto n=0; n<2*M*M; n++) CHECK(y_op1[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<2*M*M; n++) CHECK(y_op2[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<2*M*M; n++) CHECK(y_op3[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<2*M*M; n++) CHECK(y_op4[n] == doctest::Approx(y_ben[n]));
    for (auto n=0; n<2*M*M; n++) CHECK(y_op5[n] == doctest::Approx(y_ben[n]));

};

TEST_SUITE_END();

#endif // doctest
