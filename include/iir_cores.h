#ifndef IIR_CORES_H
#define IIR_CORES_H 1

#include "vectorclass.h"
#include <array>
#include "zero_init_condition.h"
#include "init_cond_correction.h"
#include "permuteV.h"



template<typename V> class IirCoreOrderTwo{

    using T = decltype(std::declval<V>().extract(0));
    static const int M = V::size();

    private:

        T _b1, _b2, _a1, _a2; // coefs

        ZeroInitCond<V> _Zic;
        InitCondCorc<V> _Icc;
        
    public:

        IirCoreOrderTwo(const T b1, const T b2, const T a1, const T a2, const T xi1=0, const T xi2=0, const T yi1=0, const T yi2=0): 
                        _b1(b1), _b2(b2), _a1(a1), _a2(a2){

                            _Zic = ZeroInitCond<V>(_b1, _b2, _a1, _a2, xi1, xi2); // homogeneous 
                            _Icc = InitCondCorc<V>(_a1, _a2, yi1, yi2); // particular

                        };

        // initialize iir core with pre-state in vector.
        IirCoreOrderTwo(const T coefs[5], const T inits[4]): _b1(coefs[1]), _b2(coefs[2]), _a1(coefs[3]), _a2(coefs[4]){

                            _Zic = ZeroInitCond<V>(coefs[1], coefs[2], coefs[3], coefs[4], inits[0], inits[1]); // homogeneous 
                            _Icc = InitCondCorc<V>(coefs[3], coefs[4], inits[2], inits[3]); // particular

                        };

        
        // scalar: accuracy check
        inline T IIR_benchmark(const T x){

            T y = x + _b1*_Zic._S[-1] + _b2*_Zic._S[-2] + _a1*_Icc._S[-1] + _a2*_Icc._S[-2];

            _Zic._S.shift(x);

            _Icc._S.shift(y);

            return y;

        };


        // block filtering
        inline V NT_ZIC_NT_ICC(const V x){

            V w = _Zic.NT_ZIC(x);

            V y = _Icc.NT_ICC(w);

            return y;

        };


        // multi-block filtering
        inline std::array<V,M> T_ZIC_T_ICC(const std::array<V,M>& x){

            std::array<V,M> x_T = _permuteV(x);

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> y_T = _Icc.T_ICC(w_T);

            std::array<V,M> y = _permuteV(y_T);

            return y;

        };


        // // multi-block filtering
        // inline std::array<V,M> T_ZIC_T_ICC2(const std::array<V,M>& x){

        //     std::array<V,M> x_T = _permuteV(x);

        //     std::array<V,M> w_T = _Zic.T_ZIC(x_T);

        //     std::array<V,M> y_T = _Icc.T_ICC2(w_T);

        //     std::array<V,M> y = _permuteV(y_T);

        //     return y;

        // };
        
        
        // // multi-block filtering
        // inline std::array<V,M> T_ZIC_T_ICC_old(const std::array<V,M>& x){

        //     std::array<V,M> x_T = _permuteV(x);

        //     std::array<V,M> w_T = _Zic.T_ZIC(x_T);

        //     std::array<V,M> y_T = _Icc.T_ICC_old(w_T);

        //     std::array<V,M> y = _permuteV(y_T);

        //     return y;

        // };


        inline V Option_1(const V x){

            V y{0};

            for (auto n=0; n<M; n++) y = mul_add(_Zic._H[n], x[n], y);

            y = mul_add(_Zic._p2, _Zic._S[-2], y);
            y = mul_add(_Zic._p1, _Zic._S[-1], y);

            y = mul_add(_Icc._h2, _Icc._S[-2], y);
            y = mul_add(_Icc._h1, _Icc._S[-1], y);

            _Zic._S.shift(x);
            _Icc._S.shift(y);

            return y;

        };


        inline std::array<V,M> Option_2(const std::array<V,M>& x){

            std::array<V,M> x_T = _permuteV(x);

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> w = _permuteV(w_T);

            std::array<V,M> y;

            for (auto n=0; n<M; n++) y[n] = _Icc.NT_ICC(w[n]);

            return y;

        };
      

        inline std::array<V,M> Option_3(const std::array<V,M>& x){

            std::array<V,M> x_T = _permuteV(x);

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> y_T = _Icc.T_ICC(w_T);

            std::array<V,M> y = _permuteV(y_T);

            return y;

        };


        // inline std::array<V,M> Option_3_2(const std::array<V,M>& x){

        //     std::array<V,M> x_T = _permuteV(x);

        //     std::array<V,M> w_T = _Zic.T_ZIC(x_T);

        //     std::array<V,M> y_T = _Icc.T_ICC2(w_T);

        //     std::array<V,M> y = _permuteV(y_T);

        //     return y;

        // };


	// below for cascaded system
        inline std::array<V,M> Mid_Option_3(const std::array<V,M>& x_T){

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> y_T = _Icc.T_ICC(w_T);

            return y_T;

        };

        inline std::array<V,M> Front_Option_3(const std::array<V,M>& x){

            std::array<V,M> x_T = _permuteV(x);

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> y_T = _Icc.T_ICC(w_T);

            return y_T;

        };


        inline std::array<V,M> End_Option_3(const std::array<V,M>& x_T){

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> y_T = _Icc.T_ICC(w_T);

            std::array<V,M> y = _permuteV(y_T);

            return y;

        };



        inline std::array<V,M> End_Option_2(const std::array<V,M>& x_T){

            std::array<V,M> w_T = _Zic.T_ZIC(x_T);

            std::array<V,M> w = _permuteV(w_T);

            std::array<V,M> y;

            for (auto n=0; n<M; n++) y[n] = _Icc.NT_ICC(w[n]);

            return y;

        };



};

#endif // header guard 
