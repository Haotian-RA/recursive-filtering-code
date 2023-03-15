/*    
    g++ -I../include -I/usr/local/include -mavx2 -mfma -I$VCL_PATH -I$EIGEN_PATH -std=c++17 -O3 -o main main.cpp
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "shift_reg.h"
#include "permuteV.h"
#include "fir_cores.h"
#include "zero_init_condition.h"
#include "init_cond_correction.h"
#include "iir_cores.h"
#include "series.h"
#include "filter.h" 
#include <chrono>
#include <iostream>
#include <cmath>
#include <complex>



template<typename U, long unsigned int N> void CoeffsFromRoots(const std::array<U,N>&, typename U::value_type []);


using V = Vec8f;
using T = decltype(std::declval<V>().extract(0));
using U = std::complex<T>;
constexpr int M = V::size();
constexpr int L = 14;


#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_SUITE_BEGIN("14th order IIR Filter:");

// the details can be checked in python notebook.


TEST_CASE("test: accuracy and stability"){

    // std::array<U,L> poles = {U((T)6/7,std::sqrt(11)/7),U((T)6/7,-std::sqrt(11)/7),
    //                         U(-0.5/5,std::sqrt(23)/5),U(-0.5/5,-std::sqrt(23)/5),
    //                         U((T)10/11,std::sqrt(17)/11),U((T)10/11,-std::sqrt(17)/11),
    //                         U((T)8/9,std::sqrt(13)/9),U((T)8/9,-std::sqrt(13)/9),
    //                         U(0.95,0),U(0.02,0),
    //                         U(-1.5/13,std::sqrt(0.8)/13),U(-1.5/13,-std::sqrt(0.8)/13),
    //                         U((T)6/7,(T)2/7),U((T)6/7,(T)(-2)/7)};

    std::array<U,L> poles = {U((T)3/7,std::sqrt(11)/7),U((T)3/7,-std::sqrt(11)/7),
                                U(-0.5/5,std::sqrt(23)/5),U(-0.5/5,-std::sqrt(23)/5),
                                U((T)(-5)/11,std::sqrt(17)/11),U((T)(-5)/11,-std::sqrt(17)/11),
                                U((T)4/9,std::sqrt(13)/9),U((T)4/9,-std::sqrt(13)/9),
                                U(0.475,0),U(-0.02,0),
                                U(0.3/13,std::sqrt(0.8)/13),U(0.3/13,-std::sqrt(0.8)/13),
                                U((T)(-3)/7,(T)2/7),U((T)(-3)/7,(T)(-2)/7)};

    T inits[L] = {2,3,5,7,-2,-3,-5,-7,1,4,6,8,-1,-4};
    T coefs[L+1];

    std::cout<<"The pre-states (yi) of 14th order iir filter are: "<<std::endl;
    for (auto n=0; n<L; n++) std::cout<<inits[n]<<std::endl;

    CoeffsFromRoots(poles,coefs);

    std::cout<<"The coefficients (a) of 14th order iir filter are: "<<std::endl;
    for (auto n=0; n<L+1; n++) std::cout<<coefs[n]<<std::endl;

    T coefs_sos[L/2][5] = {0};
    T inits_sos[L/2][4] = {0};

    for (auto n=0; n<L/2; n++){

        T tmp[3];

        std::array<U,2> poles_pair = {poles[2*n], poles[2*n+1]};
        CoeffsFromRoots(poles_pair,tmp);

        coefs_sos[n][0] = 1;
        coefs_sos[n][3] = tmp[1];
        coefs_sos[n][4] = tmp[2];

    };

    T inits_con_copy[L/2][L] = {0};
    std::copy(std::begin(inits), std::end(inits), std::begin(inits_con_copy[L/2-1]));

    for (auto n=L/2-1; n>=0; n--){

        if (n < L/2-1){
            for (auto m=0; m<2*(n+1); m++)
                inits_con_copy[n][m] = inits_con_copy[n+1][m] 
                                        - coefs_sos[n+1][3]*inits_con_copy[n+1][m+1] 
                                            - coefs_sos[n+1][4]*inits_con_copy[n+1][m+2];
        }
        
        inits_sos[n][2] = inits_con_copy[n][0];
        inits_sos[n][3] = inits_con_copy[n][1];
    
    }

    std::cout<<"The pre-states (yi) of cascaded second order iir filter are: "<<std::endl;
    for (auto n=0; n<L/2; n++){
        for (auto m=0; m<4; m++) 
            std::cout<<inits_sos[n][m]<<" ";

        std::cout<<"\n";
    }

    std::cout<<"The coefficients (a) of cascaded second order iir filter are: "<<std::endl;
    for (auto n=0; n<L/2; n++){
        for (auto m=0; m<5; m++) 
            std::cout<<coefs_sos[n][m]<<" ";

        std::cout<<"\n";
    }

    // examine accuracy, serial by hand (direct 14) vs filter passing by scalar (sos)
    std::array<T,1> in = {1};
    T out = in[0];

    for (auto n=0; n<L; n++)
        out += inits[n]*coefs[n+1];

    std::array<T,1> out2;

    Filter F(coefs_sos,inits_sos);
    auto r = F(in.begin(),in.end(),out2.begin());

    std::cout<<"The output of filter is "<<*(r-1)<<std::endl;
    
    CHECK(out2.end() == r);
    CHECK(out2[0] == doctest::Approx(out));

    // examine stability, serial by hand vs filter
    static const int v_size = 1000;
    std::array<T,v_size> v_in = {0}, v_out, v_out2;
    v_in[0] = 1;

    for (auto m=0; m<v_size; m++){

        v_out[m] = v_in[m];
        for (auto n=0; n<L; n++)
            v_out[m] += inits[n]*coefs[n+1];

        std::memmove(inits+1,inits,(L-1)*sizeof(T));
        inits[0] = v_out[m];

    };

    Filter F_v(coefs_sos,inits_sos);

    auto r_v = F_v(v_in.begin(), v_in.end(), v_out2.begin());

    for (auto n=0; n<v_size; n++) CHECK(v_out[n] == doctest::Approx(v_out[n])); // test is passed when safe poles.

    // std::cout<<"the last output of direct 14th order: "<<std::endl;
    // std::cout<<std::setprecision(10)<<v_out[v_size-1]<<std::endl; // unstable

    // std::cout<<"the last output of sos filter: "<<std::endl;
    // std::cout<<std::setprecision(10)<<v_out2[v_size-1]<<std::endl; // stable and fits the result in python notebook.

}



TEST_CASE("test: speed and perf"){

    std::array<U,L> poles = {U((T)3/7,std::sqrt(11)/7),U((T)3/7,-std::sqrt(11)/7),
                                U(-0.5/5,std::sqrt(23)/5),U(-0.5/5,-std::sqrt(23)/5),
                                U((T)(-5)/11,std::sqrt(17)/11),U((T)(-5)/11,-std::sqrt(17)/11),
                                U((T)4/9,std::sqrt(13)/9),U((T)4/9,-std::sqrt(13)/9),
                                U(0.475,0),U(-0.02,0),
                                U(0.3/13,std::sqrt(0.8)/13),U(0.3/13,-std::sqrt(0.8)/13),
                                U((T)(-3)/7,(T)2/7),U((T)(-3)/7,(T)(-2)/7)};

    std::array<U,L> zeros = {U(0.25,0.5),U(0.25,-0.5),
                             U(0.73,0.26),U(0.73,-0.26),
                             U(-0.5,0.5),U(-0.5,-0.5),
                             U(-0.73,0.52),U(-0.73,-0.52),
                             U(0.93,0),U(-0.3,0),
                             U(-0.25,0.85),U(-0.25,-0.85),
                             U(0,0.5),U(0,-0.5)};

    T coefs_sos[L/2][5] = {0};

    for (auto n=0; n<L/2; n++){

        T tmp1[3], tmp2[3];

        std::array<U,2> poles_pair = {poles[2*n], poles[2*n+1]};
        CoeffsFromRoots(poles_pair,tmp1);

        std::array<U,2> zeros_pair = {zeros[2*n], zeros[2*n+1]};
        CoeffsFromRoots(zeros_pair,tmp2);

        coefs_sos[n][0] = 1;
        coefs_sos[n][1] = -1*tmp2[1];
        coefs_sos[n][2] = -1*tmp2[2];
        coefs_sos[n][3] = tmp1[1];
        coefs_sos[n][4] = tmp1[2];

    };

    T inits_sos[L/2][4] = {0.2,0.3,0.4,0.5,
                           0.5,0.7,0.9,3,
                           -2,2.1,3,9,
                           2.2,2.3,0.5,0.3,
                           -2,-3,5,7,
                           2,3,1,8,
                           3,4,6,2};
    
    static const int vector_size = 1000000;
    std::array<T,vector_size> in = {0}, out;
    in[0] = 1;

    Filter F(coefs_sos,inits_sos);

    auto start = std::chrono::high_resolution_clock::now();

    auto r = F(in.begin(),in.end(),out.begin());

    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";


}



TEST_SUITE_END();



#endif // doctest









template<typename U, long unsigned int N> void CoeffsFromRoots(const std::array<U,N>& roots, typename U::value_type coefs[]){

    std::array<U,N+1> tmp = {0};
 
    tmp[0] = 1;
 
    for (int i=1; i<=N; i++)
		for (int j=1; j<=N-i+1; j++)
			tmp[j] += -roots[j+i-2]*tmp[j-1];

    coefs[0] = 1;
    for (auto n=1; n<N+1; n++) coefs[n] = real(U(-1,0)*tmp[n]);
 
};
























