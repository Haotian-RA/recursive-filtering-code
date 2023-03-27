#include <https://raw.githubusercontent.com/vectorclass/version2/master/vectorclass.h>
#include <array>
#include <vector>

using V = Vec16f;
using T = float;
constexpr static int N = 1;
constexpr int M = 16;

using VBlock = std::array<V, V::size()>;
alignas(256) std::array<VBlock, N> Inputs, Outputs;
alignas(256) std::array<V, V::size()> V_Inputs, V_Outputs;
V v_inputs, v_outputs;

T _b1, _b2, _a1, _a2;
V _h1, _h2;
V _rd0_22, _rd0_12, _rd0_21, _rd0_11, _rd1_22, _rd1_12, _rd1_21, _rd1_11, _rd2_22, _rd2_12, _rd2_21, _rd2_11, _rd3_22, _rd3_12, _rd3_21, _rd3_11;
V _rd4_22, _rd4_12, _rd4_21, _rd4_11;

template<typename V> class Shift{

    private:

        V _buffer{0};

    public:

        inline void shift(const T x){ 

            _buffer = blend16<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>(_buffer, x);
            
        };

        inline void shift(const V xv){ _buffer = xv; };

        inline T operator[](const int idx){ return (idx < 0) ? _buffer[M+idx] : _buffer[idx]; };

};

Shift<V> _S1, _S2, _S3;

inline std::array<V,M> T_ZIC(const std::array<V,M>& x){

    std::array<V,M> v, w;

    V xi2 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-2], _S1[-2]);
    V xi1 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-1], _S1[-1]);

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

    // _S1.shift(x[M-2][M-1]);
    // _S1.shift(x[M-1][M-1]);

    return w; 

};

inline std::array<V,M> T_ICC(const std::array<V,M>& w){

    std::array<V,M> y;

    // step 1: initialization
    y[M-2] = mul_add(_rd0_22, _S2[-2], w[M-2]);
    y[M-2] = mul_add(_rd0_12, _S2[-1], y[M-2]);
    y[M-1] = mul_add(_rd0_21, _S2[-2], w[M-1]);
    y[M-1] = mul_add(_rd0_11, _S2[-1], y[M-1]);

    // step 2: first recursion
    V tmp2 = permute16<-1,0,-1,2,-1,4,-1,6,-1,8,-1,10,-1,12,-1,14>(y[M-2]);
    V tmp1 = permute16<-1,0,-1,2,-1,4,-1,6,-1,8,-1,10,-1,12,-1,14>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd1_22, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd1_12, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd1_21, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd1_11, y[M-1]);

    // step 3: second recursion
    tmp2 = permute16<-1,-1,1,1,-1,-1,5,5,-1,-1,9,9,-1,-1,13,13>(y[M-2]);
    tmp1 = permute16<-1,-1,1,1,-1,-1,5,5,-1,-1,9,9,-1,-1,13,13>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd2_22, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd2_12, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd2_21, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd2_11, y[M-1]);

    // step 4: third recursion
    tmp2 = permute16<-1,-1,-1,-1,3,3,3,3,-1,-1,-1,-1,11,11,11,11>(y[M-2]);
    tmp1 = permute16<-1,-1,-1,-1,3,3,3,3,-1,-1,-1,-1,11,11,11,11>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd3_22, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd3_12, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd3_21, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd3_11, y[M-1]);

    // step 4: fourth recursion
    tmp2 = permute16<-1,-1,-1,-1,-1,-1,-1,-1,7,7,7,7,7,7,7,7>(y[M-2]);
    tmp1 = permute16<-1,-1,-1,-1,-1,-1,-1,-1,7,7,7,7,7,7,7,7>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd4_22, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd4_12, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd4_21, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd4_11, y[M-1]);

    // shuffle and forward
    V yi2 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(y[M-2], _S2[-2]);
    V yi1 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(y[M-1], _S2[-1]);

    for (auto i=0; i<M-2; i++){
        
        y[i] = mul_add(yi2, _h2[i], w[i]);
        y[i] = mul_add(yi1, _h1[i], y[i]);

    };

    // _S.shift(y[M-2][M-1]);
    // _S.shift(y[M-1][M-1]); 
    
    return y; 

};

inline V NT_ICC(const V w){

    V y;

    y = mul_add(_h2, _S3[-2], w);
    y = mul_add(_h1, _S3[-1], y);

    _S3.shift(y);

    return y;

};

template<typename V> inline void _permuteV16(const V matrix[16], V matrix_T[16]){

    // first round: swap lower left and upper right
    V tmp1[16];

    tmp1[0] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[0], matrix[8]);
    tmp1[1] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[1], matrix[9]);
    tmp1[2] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[2], matrix[10]);
    tmp1[3] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[3], matrix[11]);
    tmp1[4] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[4], matrix[12]);
    tmp1[5] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[5], matrix[13]);
    tmp1[6] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[6], matrix[14]);
    tmp1[7] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[7], matrix[15]);
    tmp1[8] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[0], matrix[8]);
    tmp1[9] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[1], matrix[9]);
    tmp1[10] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[2], matrix[10]);
    tmp1[11] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[3], matrix[11]);
    tmp1[12] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[4], matrix[12]);
    tmp1[13] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[5], matrix[13]);
    tmp1[14] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[6], matrix[14]);
    tmp1[15] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[7], matrix[15]);

    // second/final round: swap swap lower left and upper right within each quadrant
    V tmp2[16];

    tmp2[0] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[0], tmp1[4]);
    tmp2[1] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[1], tmp1[5]);
    tmp2[2] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[2], tmp1[6]);
    tmp2[3] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[3], tmp1[7]);
    tmp2[4] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[0], tmp1[4]);
    tmp2[5] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[1], tmp1[5]);
    tmp2[6] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[2], tmp1[6]);
    tmp2[7] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[3], tmp1[7]);
    tmp2[8] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[8], tmp1[12]);
    tmp2[9] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[9], tmp1[13]);
    tmp2[10] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[10], tmp1[14]);
    tmp2[11] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[11], tmp1[15]);
    tmp2[12] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[8], tmp1[12]);
    tmp2[13] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[9], tmp1[13]);
    tmp2[14] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[10], tmp1[14]);
    tmp2[15] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[11], tmp1[15]);

    // third/final round: swap swap lower left and upper right within each quadrant
    V tmp3[16];

    tmp3[0] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[0], tmp2[2]);
    tmp3[1] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[1], tmp2[3]);
    tmp3[2] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[0], tmp2[2]);
    tmp3[3] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[1], tmp2[3]);
    tmp3[4] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[4], tmp2[6]);
    tmp3[5] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[5], tmp2[7]);
    tmp3[6] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[4], tmp2[6]);
    tmp3[7] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[5], tmp2[7]);
    tmp3[8] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[8], tmp2[10]);
    tmp3[9] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[9], tmp2[11]);
    tmp3[10] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[8], tmp2[10]);
    tmp3[11] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[9], tmp2[11]);
    tmp3[12] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[12], tmp2[14]);
    tmp3[13] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[13], tmp2[15]);
    tmp3[14] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[12], tmp2[14]);
    tmp3[15] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[13], tmp2[15]);

    // third/final round: swap elements anti-diagonally adjacent
    matrix_T[0] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[0], tmp3[1]);
    matrix_T[1] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[0], tmp3[1]);
    matrix_T[2] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[2], tmp3[3]);
    matrix_T[3] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[2], tmp3[3]);
    matrix_T[4] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[4], tmp3[5]);
    matrix_T[5] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[4], tmp3[5]);
    matrix_T[6] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[6], tmp3[7]);
    matrix_T[7] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[6], tmp3[7]); 
    matrix_T[8] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[8], tmp3[9]);
    matrix_T[9] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[8], tmp3[9]);
    matrix_T[10] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[10], tmp3[11]);
    matrix_T[11] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[10], tmp3[11]);
    matrix_T[12] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[12], tmp3[13]);
    matrix_T[13] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[12], tmp3[13]);
    matrix_T[14] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[14], tmp3[15]);
    matrix_T[15] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[14], tmp3[15]); 

}

// do transpose of matrices in Vec4f by 4 or Vec8f by 8.
inline std::array<V,V::size()> _permuteV(const std::array<V,V::size()>& matrix){

    std::array<V,V::size()> matrix_T;
    
    _permuteV16(matrix.data(), matrix_T.data());

    return matrix_T;

}

inline std::array<V,M> Front_Option_3(const std::array<V,M>& x){

    std::array<V,M> x_T = _permuteV(x);

    std::array<V,M> w_T = T_ZIC(x_T);

    std::array<V,M> y_T = T_ICC(w_T);

    return y_T;

};


inline std::array<V,M> End_Option_2(const std::array<V,M>& x_T){

    std::array<V,M> w_T = T_ZIC(x_T);

    std::array<V,M> w = _permuteV(w_T);

    std::array<V,M> y;

    for (auto n=0; n<M; n++) y[n] = NT_ICC(w[n]);

    return y;

};

int main(){

    for (auto n=0; n<N; n++) {

        Outputs[n] = End_Option_2(Front_Option_3(Inputs[n]));

    }

    return 0;

}


