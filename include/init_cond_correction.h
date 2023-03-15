#ifndef INIT_COND_CORRECTION_H
#define INIT_COND_CORRECTION_H 1

#include "shift_reg.h"
#include <array>
#include "vectorclass.h"

template<typename V> class InitCondCorc{

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size();

    private:

        T _a1, _a2; 
        V _h_22, _h_12, _h_21, _h_11;
        V _rd0_22, _rd0_12, _rd0_21, _rd0_11; // vectors for RD
        V _rd1_22, _rd1_12, _rd1_21, _rd1_11; 
        V _rd2_22, _rd2_12, _rd2_21, _rd2_11;
        V _rd3_22, _rd3_12, _rd3_21, _rd3_11;
        std::array<V,M> _T_22, _T_12, _T_21, _T_11;

    public:

        Shift<V> _S;
        V _h1, _h2;


        InitCondCorc(){};

        InitCondCorc(const T a1, const T a2, const T yi1=0, const T yi2=0): _a1(a1), _a2(a2){ 

            _S.shift(yi2);
            _S.shift(yi1);

            impulse_response();

            T_ICC_transit_matrices();

            T_ICC_RD_vectors();

        };


        inline void impulse_response(){

            T h0[M+1];
            
            h0[0] = 1;
            h0[1] = _a1;

            for (auto i=2; i<M+1; i++) h0[i] = _a1*h0[i-1] + _a2*h0[i-2];

            _h1.load(&h0[1]);
            _h2.load(&h0[0]);

            _h2 *= _a2;

        };


        inline void T_icc_impulse_response(){ // pre-compute the four elements of C and power of C
        
            // C = [h_22 h_12; h_21 h_11]
            T h_22[M] = {0}, h_12[M] = {0}, h_21[M] = {0}, h_11[M] = {0}; // h_xy stands for _hx[M-y] in the above impulse response

            h_22[0] = _h2[M-2];
            h_12[0] = _h1[M-2];
            h_21[0] = _h2[M-1];
            h_11[0] = _h1[M-1];

            for (auto n=1; n<M; n++){

                h_22[n] = _h2[M-2]*h_22[n-1] + _h2[M-1]*h_12[n-1];
                h_12[n] = _h1[M-2]*h_22[n-1] + _h1[M-1]*h_12[n-1];
                h_21[n] = _h2[M-2]*h_21[n-1] + _h2[M-1]*h_11[n-1];
                h_11[n] = _h1[M-2]*h_21[n-1] + _h1[M-1]*h_11[n-1];

            }

            _h_22.load(&h_22[0]);
            _h_12.load(&h_12[0]);
            _h_21.load(&h_21[0]);
            _h_11.load(&h_11[0]);

        };


        inline void T_ICC_transit_matrices(){

            T_icc_impulse_response();


            if constexpr (M == 4){

                _T_22[0] = blend4<4,0,1,2>(_h_22,1);
                _T_12[0] = permute4<-1,0,1,2>(_h_12);
                _T_21[0] = permute4<-1,0,1,2>(_h_21);
                _T_11[0] = blend4<4,0,1,2>(_h_11,1);

                for (auto n=1; n<M; n++){

                    _T_22[n] = permute4<-1,0,1,2>(_T_22[n-1]);

                    _T_12[n] = permute4<-1,0,1,2>(_T_12[n-1]);

                    _T_21[n] = permute4<-1,0,1,2>(_T_21[n-1]);

                    _T_11[n] = permute4<-1,0,1,2>(_T_11[n-1]);

                }

            }


            if constexpr (M == 8){

                _T_22[0] = blend8<8,0,1,2,3,4,5,6>(_h_22,1);
                _T_12[0] = permute8<-1,0,1,2,3,4,5,6>(_h_12);
                _T_21[0] = permute8<-1,0,1,2,3,4,5,6>(_h_21);
                _T_11[0] = blend8<8,0,1,2,3,4,5,6>(_h_11,1);

                for (auto n=1; n<M; n++){

                    _T_22[n] = permute8<-1,0,1,2,3,4,5,6>(_T_22[n-1]);

                    _T_12[n] = permute8<-1,0,1,2,3,4,5,6>(_T_12[n-1]);

                    _T_21[n] = permute8<-1,0,1,2,3,4,5,6>(_T_21[n-1]);

                    _T_11[n] = permute8<-1,0,1,2,3,4,5,6>(_T_11[n-1]);

                }

            }



        };


        inline void T_ICC_RD_vectors(){

            T_icc_impulse_response();

            if constexpr (M == 4){

                // RD initialization
                _rd0_22 = permute4<0,-1,-1,-1>(_h_22); // RD level 0, h_22
                _rd0_12 = permute4<0,-1,-1,-1>(_h_12);
                _rd0_21 = permute4<0,-1,-1,-1>(_h_21);
                _rd0_11 = permute4<0,-1,-1,-1>(_h_11);

                // RD recursion 1
                _rd1_22 = permute4<-1,0,-1,0>(_h_22);
                _rd1_12 = permute4<-1,0,-1,0>(_h_12);
                _rd1_21 = permute4<-1,0,-1,0>(_h_21);
                _rd1_11 = permute4<-1,0,-1,0>(_h_11);

                // RD recursion 2
                _rd2_22 = permute4<-1,-1,0,1>(_h_22);
                _rd2_12 = permute4<-1,-1,0,1>(_h_12);
                _rd2_21 = permute4<-1,-1,0,1>(_h_21);
                _rd2_11 = permute4<-1,-1,0,1>(_h_11);

            };


            if constexpr (M == 8){

                // RD initialization
                _rd0_22 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_22); 
                _rd0_12 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_12);
                _rd0_21 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_21);
                _rd0_11 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_11);

                // RD recursion 1
                _rd1_22 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_22);
                _rd1_12 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_12);
                _rd1_21 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_21);
                _rd1_11 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_11);

                // RD recursion 2
                _rd2_22 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_22);
                _rd2_12 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_12);
                _rd2_21 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_21);
                _rd2_11 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_11);

                // RD recursion 3
                _rd3_22 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_22);
                _rd3_12 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_12);
                _rd3_21 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_21);
                _rd3_11 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_11);

            };

        };


        inline V NT_ICC(const V w){

            V y;

            y = mul_add(_h2, _S[-2], w);
            y = mul_add(_h1, _S[-1], y);

            _S.shift(y);

            return y;

        };
        
        
        inline std::array<V,M> T_ICC_old(const std::array<V,M>& w){ // big matrix multiplication

            std::array<V,M> y{0};
            V yi2, yi1;

            for (auto n=0; n<M; n++){

                y[M-2] = mul_add(_T_22[n], w[M-2][n], y[M-2]);
                y[M-2] = mul_add(_T_12[n], w[M-1][n], y[M-2]);
    
            }

            y[M-2] = mul_add(_h_22, _S[-2], y[M-2]);
            y[M-2] = mul_add(_h_12, _S[-1], y[M-2]);

            for (auto n=0; n<M; n++){

                y[M-1] = mul_add(_T_21[n], w[M-2][n], y[M-1]);
                y[M-1] = mul_add(_T_11[n], w[M-1][n], y[M-1]);

            }

            y[M-1] = mul_add(_h_21, _S[-2], y[M-1]);
            y[M-1] = mul_add(_h_11, _S[-1], y[M-1]);

            if constexpr (M == 4){

                yi2 = blend4<4,0,1,2>(y[M-2], _S[-2]);
                yi1 = blend4<4,0,1,2>(y[M-1], _S[-1]);
                
            }

            if constexpr (M == 8){

                yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
                yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);
                
            }

            for (auto i=0; i<M-2; i++){
                
                y[i] = mul_add(yi2, _h2[i], w[i]);
                y[i] = mul_add(yi1, _h1[i], y[i]);

            };
     
            _S.shift(y[M-2][M-1]);
            _S.shift(y[M-1][M-1]); 
            
            return y;

        };


        inline std::array<V,M> T_ICC(const std::array<V,M>& w){ // recursive doubling tree 1

            std::array<V,M> y;
            V tmp2, tmp1, yi2, yi1;

            // step 1: initialization
            y[M-2] = mul_add(_rd0_22, _S[-2], w[M-2]);
            y[M-2] = mul_add(_rd0_12, _S[-1], y[M-2]);
            y[M-1] = mul_add(_rd0_21, _S[-2], w[M-1]);
            y[M-1] = mul_add(_rd0_11, _S[-1], y[M-1]);
            

            if constexpr (M == 4){

                // step 2: first recursion
                tmp2 = permute4<-1,0,-1,2>(y[M-2]);
                tmp1 = permute4<-1,0,-1,2>(y[M-1]);

                y[M-2] = mul_add(tmp2, _rd1_22, y[M-2]);
                y[M-2] = mul_add(tmp1, _rd1_12, y[M-2]);
                y[M-1] = mul_add(tmp2, _rd1_21, y[M-1]);
                y[M-1] = mul_add(tmp1, _rd1_11, y[M-1]);

                // step 3: second recursion
                tmp2 = permute4<-1,-1,1,1>(y[M-2]);
                tmp1 = permute4<-1,-1,1,1>(y[M-1]);

                y[M-2] = mul_add(tmp2, _rd2_22, y[M-2]);
                y[M-2] = mul_add(tmp1, _rd2_12, y[M-2]);
                y[M-1] = mul_add(tmp2, _rd2_21, y[M-1]);
                y[M-1] = mul_add(tmp1, _rd2_11, y[M-1]);

                // shuffle and forward
                yi2 = blend4<4,0,1,2>(y[M-2], _S[-2]);
                yi1 = blend4<4,0,1,2>(y[M-1], _S[-1]);

            };


            if constexpr (M == 8){ // less cycles. no extra broadcast cost

                // step 2: first recursion
                tmp2 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-2]);
                tmp1 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-1]);

                y[M-2] = mul_add(tmp2, _rd1_22, y[M-2]);
                y[M-2] = mul_add(tmp1, _rd1_12, y[M-2]);
                y[M-1] = mul_add(tmp2, _rd1_21, y[M-1]);
                y[M-1] = mul_add(tmp1, _rd1_11, y[M-1]);

                // step 3: second recursion
                tmp2 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-2]);
                tmp1 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-1]);

                y[M-2] = mul_add(tmp2, _rd2_22, y[M-2]);
                y[M-2] = mul_add(tmp1, _rd2_12, y[M-2]);
                y[M-1] = mul_add(tmp2, _rd2_21, y[M-1]);
                y[M-1] = mul_add(tmp1, _rd2_11, y[M-1]);

                // step 4: third recursion
                tmp2 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-2]);
                tmp1 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-1]);

                y[M-2] = mul_add(tmp2, _rd3_22, y[M-2]);
                y[M-2] = mul_add(tmp1, _rd3_12, y[M-2]);
                y[M-1] = mul_add(tmp2, _rd3_21, y[M-1]);
                y[M-1] = mul_add(tmp1, _rd3_11, y[M-1]);

                // shuffle and forward
                yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
                yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);

            };


            for (auto i=0; i<M-2; i++){
                
                y[i] = mul_add(yi2, _h2[i], w[i]);
                y[i] = mul_add(yi1, _h1[i], y[i]);

            };
     
            _S.shift(y[M-2][M-1]);
            _S.shift(y[M-1][M-1]); 
            
            return y; 

        };


        inline std::array<V,M> T_ICC2(const std::array<V,M>& w){ // recursive doubling tree 2

            std::array<V,M> y;
            V tmp2{0}, tmp1{0}, yi2, yi1;
            tmp2.insert(0,_S[-2]);
            tmp1.insert(0,_S[-1]);
            
            // step 1: initialization
            y[M-2] = mul_add(tmp2, _h_22[0], w[M-2]);
            y[M-2] = mul_add(tmp1, _h_12[0], y[M-2]);
            y[M-1] = mul_add(tmp2, _h_21[0], w[M-1]);
            y[M-1] = mul_add(tmp1, _h_11[0], y[M-1]);


            if constexpr (M == 4){

                // step 2: first recursion
                tmp2 = permute4<-1,0,1,2>(y[M-2]);
                tmp1 = permute4<-1,0,1,2>(y[M-1]);

                y[M-2] = mul_add(tmp2, _h_22[0], y[M-2]);
                y[M-2] = mul_add(tmp1, _h_12[0], y[M-2]);
                y[M-1] = mul_add(tmp2, _h_21[0], y[M-1]);
                y[M-1] = mul_add(tmp1, _h_11[0], y[M-1]);

                // step 3: second recursion
                tmp2 = permute4<-1,-1,0,1>(y[M-2]);
                tmp1 = permute4<-1,-1,0,1>(y[M-1]);

                y[M-2] = mul_add(tmp2, _h_22[1], y[M-2]);
                y[M-2] = mul_add(tmp1, _h_12[1], y[M-2]);
                y[M-1] = mul_add(tmp2, _h_21[1], y[M-1]);
                y[M-1] = mul_add(tmp1, _h_11[1], y[M-1]);

                // shuffle and forward
                yi2 = blend4<4,0,1,2>(y[M-2], _S[-2]);
                yi1 = blend4<4,0,1,2>(y[M-1], _S[-1]);

            };


            if constexpr (M == 8){

                // step 2: first recursion
                tmp2 = permute8<-1,0,1,2,3,4,5,6>(y[M-2]);
                tmp1 = permute8<-1,0,1,2,3,4,5,6>(y[M-1]);

                y[M-2] = mul_add(tmp2, _h_22[0], y[M-2]);
                y[M-2] = mul_add(tmp1, _h_12[0], y[M-2]);
                y[M-1] = mul_add(tmp2, _h_21[0], y[M-1]);
                y[M-1] = mul_add(tmp1, _h_11[0], y[M-1]);

                // step 3: second recursion
                tmp2 = permute8<-1,-1,0,1,2,3,4,5>(y[M-2]);
                tmp1 = permute8<-1,-1,0,1,2,3,4,5>(y[M-1]);

                y[M-2] = mul_add(tmp2, _h_22[1], y[M-2]);
                y[M-2] = mul_add(tmp1, _h_12[1], y[M-2]);
                y[M-1] = mul_add(tmp2, _h_21[1], y[M-1]);
                y[M-1] = mul_add(tmp1, _h_11[1], y[M-1]);

                // step 4: third recursion
                tmp2 = permute8<-1,-1,-1,-1,0,1,2,3>(y[M-2]);
                tmp1 = permute8<-1,-1,-1,-1,0,1,2,3>(y[M-1]);

                y[M-2] = mul_add(tmp2, _h_22[3], y[M-2]);
                y[M-2] = mul_add(tmp1, _h_12[3], y[M-2]);
                y[M-1] = mul_add(tmp2, _h_21[3], y[M-1]);
                y[M-1] = mul_add(tmp1, _h_11[3], y[M-1]);

                // shuffle and forward
                yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
                yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);

            };


            for (auto i=0; i<M-2; i++){
                
                y[i] = mul_add(yi2, _h2[i], w[i]);
                y[i] = mul_add(yi1, _h1[i], y[i]);

            };
     
            _S.shift(y[M-2][M-1]);
            _S.shift(y[M-1][M-1]); 
            
            return y; 

        };







};

#endif // header guard 










