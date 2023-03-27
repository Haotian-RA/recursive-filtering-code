#ifndef ZERO_INIT_CONDITION_H
#define ZERO_INIT_CONDITION_H 1


#include <array>
#include "vectorclass.h"
#include "shift_reg.h"


template<typename V> class ZeroInitCond{

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size();

    private:

        T _b1, _b2, _a1, _a2; 


    public:

        Shift<V> _S;
        V _p1, _p2, _h1, _h2;

        std::array<V,M> _H;


        ZeroInitCond(){};

        ZeroInitCond(const T b1, const T b2, const T a1, const T a2, const T xi1=0, const T xi2=0): _b1(b1), _b2(b2), _a1(a1), _a2(a2){

            _S.shift(xi2);
            _S.shift(xi1);

            NT_ZIC_transition_matrix();

        };





        inline void particular_impulse_response(){

            T p1[M], p2[M];

            p1[0] = _b1;
            p1[1] = _a1*_b1 + _b2;
            p2[0] = _b2;
            p2[1] = _a1*_b2;

            for (auto i=2; i<M+1; i++){

                p1[i] = _a1*p1[i-1] + _a2*p1[i-2];
                p2[i] = _a1*p2[i-1] + _a2*p2[i-2];

            }

            _p1.load(&p1[0]);
            _p2.load(&p2[0]);

        }

        inline void homogeneous_impulse_response(){

            T h0[M+1];
            
            h0[0] = 1;
            h0[1] = _a1;

            for (auto i=2; i<M+1; i++) h0[i] = _a1*h0[i-1] + _a2*h0[i-2];

            _h1.load(&h0[1]);
            _h2.load(&h0[0]);

            _h2 *= _a2;

        }

        inline void NT_ZIC_transition_matrix(){

            particular_impulse_response();

            homogeneous_impulse_response();

            V tmp;

            if constexpr (M == 4){

                tmp = blend4<4,0,1,2>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){

                    _H[n] = tmp;

                    tmp = permute4<-1,0,1,2>(tmp);

                }


            }

            if constexpr (M == 8){ 
                
                tmp = blend8<8,0,1,2,3,4,5,6>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){

                    _H[n] = tmp;

                    tmp = permute8<-1,0,1,2,3,4,5,6>(tmp);

                }

            }

            
            if constexpr (M == 16){ 
                
                tmp = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){

                    _H[n] = tmp;

                    tmp = permute16<-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(tmp);

                }

            }

        };


        inline V NT_ZIC(const V x){

            V w{0};

            for (auto n=0; n<M; n++) w = mul_add(_H[n], x[n], w);

            w = mul_add(_p2, _S[-2], w);
            w = mul_add(_p1, _S[-1], w);

            _S.shift(x);

            return w; 

        };


        inline std::array<V,M> T_ZIC(const std::array<V,M>& x){

            std::array<V,M> v, w;

            V xi2, xi1;

            if constexpr (M == 4){

                xi2 = blend4<4,0,1,2>(x[M-2], _S[-2]);
                xi1 = blend4<4,0,1,2>(x[M-1], _S[-1]);

            }

            if constexpr (M == 8){

                xi2 = blend8<8,0,1,2,3,4,5,6>(x[M-2], _S[-2]);
                xi1 = blend8<8,0,1,2,3,4,5,6>(x[M-1], _S[-1]);

            }

            if constexpr (M == 16){

                xi2 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-2], _S[-2]);
                xi1 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-1], _S[-1]);

            }

            v[0] = mul_add(xi2, _b2, x[0]);
            v[0] = mul_add(xi1, _b1, v[0]);

            w[0] = v[0];

            v[1] = mul_add(xi1, _b2, x[1]);
            v[1] = mul_add(x[0], _b1, v[1]);

            w[1] = mul_add(v[0], _a1, v[1]);

            for (auto n=2; n<M; n++){

                v[n] = mul_add(x[n-2], _b2, x[n]);
                v[n] = mul_add(x[n-1], _b1, v[n]);

                w[n] = mul_add(w[n-2], _a2, v[n]);
                w[n] = mul_add(w[n-1], _a1, w[n]);

            }

            _S.shift(x[M-2][M-1]);
            _S.shift(x[M-1][M-1]);

            return w; 

        };
        
};



#endif // header guard 
