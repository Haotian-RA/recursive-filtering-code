/*    
    clang++ -I/usr/local/include -mavx2 -mfma -I$VCL_PATH -std=c++20 -O3 -o main main.cpp
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "shift_reg.h"
#include "iir_cores.h"
#include "series.h"
#include "filter.h"


// checking accuracy of each function and option


#ifdef DOCTEST_LIBRARY_INCLUDED


#include <numeric>

TEST_CASE("IIR second core accuracy test, M=8:"){

    using V = Vec8f;
    const int M = V::size();
    using T = float;

    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 


    // benchmark (scalar)
    IirCoreOrderTwo<V> I1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y_ben;

    for (auto n=0; n<M*M; n++) y_ben[n] = I1.IIR_benchmark(data[n]);

    std::array<V,M> x, y;

    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);
    

    // NT_ZIC and NT_ICC
    IirCoreOrderTwo<V> I2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y2;

    for (auto n=0; n<M; n++) y[n] = I2.NT_ZIC_NT_ICC(x[n]);

    for (auto n=0; n<M; n++) y[n].store(&y2[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y2[n] == doctest::Approx(y_ben[n]));



    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y3;

    y = I3.T_ZIC_T_ICC(x);

    for (auto n=0; n<M; n++) y[n].store(&y3[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y3[n] == doctest::Approx(y_ben[n]));


    // Option 2: T_ZIC and NT_ICC
    IirCoreOrderTwo<V> I4(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y4;

    y = I4.Option_2(x);

    for (auto n=0; n<M; n++) y[n].store(&y4[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y4[n] == doctest::Approx(y_ben[n]));


    // Option 3: T_ZIC and T_ICC
    IirCoreOrderTwo<V> I5(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y5;

    y = I5.Option_3(x);

    for (auto n=0; n<M; n++) y[n].store(&y5[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y5[n] == doctest::Approx(y_ben[n]));



    // Option 1: NT_ZIC and NT_ICC
    IirCoreOrderTwo<V> I6(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y6;

    for (auto n=0; n<M; n++) y[n] = I6.Option_1(x[n]);

    for (auto n=0; n<M; n++) y[n].store(&y6[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y6[n] == doctest::Approx(y_ben[n]));


    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I7(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y7;

    y = I7.T_ZIC_T_ICC2(x);

    for (auto n=0; n<M; n++) y[n].store(&y7[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y7[n] == doctest::Approx(y_ben[n]));
    
    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I8(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y8;

    y = I8.T_ZIC_T_ICC_old(x);

    for (auto n=0; n<M; n++) y[n].store(&y8[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y8[n] == doctest::Approx(y_ben[n]));


}


TEST_CASE("IIR second core accuracy test, M=4:"){

    using V = Vec4f;
    const int M = V::size();
    using T = float;

    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 


    // benchmark (scalar)
    IirCoreOrderTwo<V> I1(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y_ben;

    for (auto n=0; n<M*M; n++) y_ben[n] = I1.IIR_benchmark(data[n]);

    std::array<V,M> x, y;

    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);


    // NT_ZIC and NT_ICC
    IirCoreOrderTwo<V> I2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y2;

    for (auto n=0; n<M; n++) y[n] = I2.NT_ZIC_NT_ICC(x[n]);

    for (auto n=0; n<M; n++) y[n].store(&y2[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y2[n] == doctest::Approx(y_ben[n]));



    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y3;

    y = I3.T_ZIC_T_ICC(x);

    for (auto n=0; n<M; n++) y[n].store(&y3[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y3[n] == doctest::Approx(y_ben[n]));


    // Option 2: T_ZIC and NT_ICC
    IirCoreOrderTwo<V> I4(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y4;

    y = I4.Option_2(x);

    for (auto n=0; n<M; n++) y[n].store(&y4[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y4[n] == doctest::Approx(y_ben[n]));


    // Option 3: T_ZIC and T_ICC
    IirCoreOrderTwo<V> I5(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y5;

    y = I5.Option_3(x);

    for (auto n=0; n<M; n++) y[n].store(&y5[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y5[n] == doctest::Approx(y_ben[n]));



    // Option 1: NT_ZIC and NT_ICC
    IirCoreOrderTwo<V> I6(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y6;

    for (auto n=0; n<M; n++) y[n] = I6.Option_1(x[n]);

    for (auto n=0; n<M; n++) y[n].store(&y6[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y6[n] == doctest::Approx(y_ben[n]));


    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I7(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y7;

    y = I7.T_ZIC_T_ICC2(x);

    for (auto n=0; n<M; n++) y[n].store(&y7[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y7[n] == doctest::Approx(y_ben[n]));


    // T_ZIC and T_ICC
    IirCoreOrderTwo<V> I8(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y8;

    y = I8.T_ZIC_T_ICC_old(x);

    for (auto n=0; n<M; n++) y[n].store(&y8[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y8[n] == doctest::Approx(y_ben[n]));



}


TEST_CASE("Cascaded options, M=8:"){

    using V = Vec8f;
    const int M = V::size();
    using T = float;

    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0); 


    // benchmark (scalar)
    IirCoreOrderTwo<V> I1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y_ben;

    for (auto n=0; n<M*M; n++) y_ben[n] = I2.IIR_benchmark(I1.IIR_benchmark(data[n]));

    std::array<V,M> x, y;

    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);


    // Option 2 + Option 2
    IirCoreOrderTwo<V> I3(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I4(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y4;

    y = I4.Option_2(I3.Option_2(x));

    for (auto n=0; n<M; n++) y[n].store(&y4[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y4[n] == doctest::Approx(y_ben[n]));


    // Option 3 + Option 2
    IirCoreOrderTwo<V> I5(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I6(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y6;

    y = I6.End_Option_2(I5.Front_Option_3(x));

    for (auto n=0; n<M; n++) y[n].store(&y6[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y6[n] == doctest::Approx(y_ben[n]));



    // Option 3 + Option 3
    IirCoreOrderTwo<V> I7(b1,b2,a1,a2,xi1,xi2,yi1,yi2),I8(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y8;

    y = I8.End_Option_3(I7.Front_Option_3(x));

    for (auto n=0; n<M; n++) y[n].store(&y8[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y8[n] == doctest::Approx(y_ben[n]));





}


TEST_CASE("Series accuracy test:"){

    using V = Vec8f;
    const int M = V::size();
    using T = float;

    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::vector<T> data(M*M);
    std::iota(data.begin(), data.end(), 0);  

    // baseline (scalar)
    IirCoreOrderTwo<V> I1(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I2(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, M*M> y_ben;

    for (auto n=0; n<M*M; n++) y_ben[n] = I3.IIR_benchmark(I2.IIR_benchmark(I1.IIR_benchmark(data[n])));

    std::array<V,M> x, y;

    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    // series
    IirCoreOrderTwo<V> I4(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I5(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I6(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    Series<IirCoreOrderTwo<V>, IirCoreOrderTwo<V>, IirCoreOrderTwo<V>> S1(I4, I5, I6);

    std::array<T, M*M> y1;

    y = S1(x);

    for (auto n=0; n<M; n++) y[n].store(&y1[n*M]); 

    for (auto n=0; n<M*M; n++) CHECK(y1[n] == doctest::Approx(y_ben[n]));



    // series from make
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};

    auto S2 = series_from_coeffs<T,V>(coefs, inits);

    std::array<T, M*M> y2;

    y = S2(x);

    for (auto n=0; n<M; n++) y[n].store(&y2[n*M]);

    for (auto n=0; n<M*M; n++) CHECK(y2[n] == doctest::Approx(y_ben[n]));



};



TEST_CASE("Filter accuracy test:"){

    using V = Vec8f;
    const int M = V::size();
    using T = float;

    T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3;
    T xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::vector<T> data(2*M*M);
    std::iota(data.begin(), data.end(), 0);  

    // baseline (scalar)
    IirCoreOrderTwo<V> I1(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I2(b1,b2,a1,a2,xi1,xi2,yi1,yi2), I3(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T, 2*M*M> y_ben;

    for (auto n=0; n<M*M; n++) y_ben[n] = I3.IIR_benchmark(I2.IIR_benchmark(I1.IIR_benchmark(data[n])));

    for (auto n=M*M; n<2*M*M; n++) y_ben[n] = I3.IIR_benchmark(I2.IIR_benchmark(I1.IIR_benchmark(data[n])));

    std::array<T,2*M*M> x;

    for (auto n=0; n<2*M*M; n++) x[n] = data[n];
   
    // default filter
    T coefs[3][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[3][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};

    Filter F0(coefs,inits);

    std::array<T, 2*M*M> y0;

    F0(x.begin(),x.end(),y0.begin());

    for (auto n=0; n<2*M*M; n++) CHECK(y0[n] == doctest::Approx(y_ben[n]));

    // filter 1
    Filter F1(coefs,inits);

    std::array<T, 2*M*M> y1;

    F1.Cascaded_option_1(x.begin(),x.end(),y1.begin());

    for (auto n=0; n<2*M*M; n++) CHECK(y1[n] == doctest::Approx(y_ben[n]));

    // filter 2
    Filter F2(coefs,inits);

    std::array<T, 2*M*M> y2;

    F2.Cascaded_option_2(x.begin(),x.end(),y2.begin());

    for (auto n=0; n<2*M*M; n++) CHECK(y2[n] == doctest::Approx(y_ben[n]));

    // filter 3
    Filter F3(coefs,inits);

    std::array<T, 2*M*M> y3;

    F3.Cascaded_option_3(x.begin(),x.end(),y3.begin());

    for (auto n=0; n<2*M*M; n++) CHECK(y3[n] == doctest::Approx(y_ben[n]));

};


TEST_SUITE_END();


#endif // doctest












