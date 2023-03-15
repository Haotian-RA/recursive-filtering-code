#include <https://raw.githubusercontent.com/vectorclass/version2/master/vectorclass.h>
#include <array>
#include <vector>

using V = Vec8f;
using T = float;
constexpr int M1 = 8;
constexpr static int N = 1;
constexpr int M = 8;

using VBlock = std::array<V, M1>;
alignas(64) std::array<VBlock, N> Inputs, Outputs;
alignas(64) std::array<V, M1> V_Inputs, V_Outputs;
V v_inputs, v_outputs;

T _b1, _b2, _a1, _a2;
V _h1, _h2;
V _rd0_tl, _rd0_tr, _rd0_bl, _rd0_br, _rd1_tl, _rd1_tr, _rd1_bl, _rd1_br, _rd2_tl, _rd2_tr, _rd2_bl, _rd2_br, _rd3_tl, _rd3_tr, _rd3_bl, _rd3_br;

template<typename V> class Shift{

    private:

        V _buffer{0};

    public:

        inline void shift(const T x){ 

            _buffer = blend8<1,2,3,4,5,6,7,8>(_buffer, x);
            
        };

        inline void shift(const V xv){ _buffer = xv; };

        inline T operator[](const int idx){ return (idx < 0) ? _buffer[M+idx] : _buffer[idx]; };

};

Shift<V> _S1, _S2;

inline std::array<V,M> T_ZIC(const std::array<V,M>& x){

    std::array<V,M> v, w;

    V    xi2 = blend8<8,0,1,2,3,4,5,6>(x[M-2], _S1[-2]);
    V    xi1 = blend8<8,0,1,2,3,4,5,6>(x[M-1], _S1[-1]);

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

    _S1.shift(x[M-2][M-1]);
    _S1.shift(x[M-1][M-1]);

    return w; 

};

inline std::array<V,M> T_ICC(const std::array<V,M>& w){

    std::array<V,M> y;

    // step 1: initialization
    y[M-2] = mul_add(_rd0_tl, _S2[-2], w[M-2]);
    y[M-2] = mul_add(_rd0_bl, _S2[-1], y[M-2]);
    y[M-1] = mul_add(_rd0_tr, _S2[-2], w[M-1]);
    y[M-1] = mul_add(_rd0_br, _S2[-1], y[M-1]);

    V tmp2, tmp1, yi2, yi1;

    // step 2: first recursion
    tmp2 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-2]);
    tmp1 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd1_tl, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd1_bl, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd1_tr, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd1_br, y[M-1]);

    // step 3: second recursion
    tmp2 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-2]);
    tmp1 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd2_tl, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd2_bl, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd2_tr, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd2_br, y[M-1]);

    // step 4: third recursion
    tmp2 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-2]);
    tmp1 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-1]);

    y[M-2] = mul_add(tmp2, _rd3_tl, y[M-2]);
    y[M-2] = mul_add(tmp1, _rd3_bl, y[M-2]);
    y[M-1] = mul_add(tmp2, _rd3_tr, y[M-1]);
    y[M-1] = mul_add(tmp1, _rd3_br, y[M-1]);

    // shuffle and forward
    yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S2[-2]);
    yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S2[-1]);

    for (auto i=0; i<M-2; i++){
        
        y[i] = mul_add(yi2, _h2[i], w[i]);
        y[i] = mul_add(yi1, _h1[i], y[i]);

    };

    _S1.shift(y[M-2][M-1]);
    _S1.shift(y[M-1][M-1]); 
    
    return y; 

};

inline void _permuteV8(const V matrix[8], V matrix_T[8]){

    // first round: swap lower left and upper right
    V tmp1[8];

    tmp1[0] = blend8<0,1,2,3,8,9,10,11>(matrix[0], matrix[4]);
    tmp1[1] = blend8<0,1,2,3,8,9,10,11>(matrix[1], matrix[5]);
    tmp1[2] = blend8<0,1,2,3,8,9,10,11>(matrix[2], matrix[6]);
    tmp1[3] = blend8<0,1,2,3,8,9,10,11>(matrix[3], matrix[7]);
    tmp1[4] = blend8<4,5,6,7,12,13,14,15>(matrix[0], matrix[4]);
    tmp1[5] = blend8<4,5,6,7,12,13,14,15>(matrix[1], matrix[5]);
    tmp1[6] = blend8<4,5,6,7,12,13,14,15>(matrix[2], matrix[6]);
    tmp1[7] = blend8<4,5,6,7,12,13,14,15>(matrix[3], matrix[7]);

    // second/final round: swap swap lower left and upper right within each quadrant
    V tmp2[8];

    tmp2[0] = blend8<0,1,8,9,4,5,12,13>(tmp1[0], tmp1[2]);
    tmp2[1] = blend8<0,1,8,9,4,5,12,13>(tmp1[1], tmp1[3]);
    tmp2[2] = blend8<2,3,10,11,6,7,14,15>(tmp1[0], tmp1[2]);
    tmp2[3] = blend8<2,3,10,11,6,7,14,15>(tmp1[1], tmp1[3]);
    tmp2[4] = blend8<0,1,8,9,4,5,12,13>(tmp1[4], tmp1[6]);
    tmp2[5] = blend8<0,1,8,9,4,5,12,13>(tmp1[5], tmp1[7]);
    tmp2[6] = blend8<2,3,10,11,6,7,14,15>(tmp1[4], tmp1[6]);
    tmp2[7] = blend8<2,3,10,11,6,7,14,15>(tmp1[5], tmp1[7]);

    // third/final round: swap elements anti-diagonally adjacent
    matrix_T[0] = blend8<0,8,2,10,4,12,6,14>(tmp2[0], tmp2[1]);
    matrix_T[1] = blend8<1,9,3,11,5,13,7,15>(tmp2[0], tmp2[1]);
    matrix_T[2] = blend8<0,8,2,10,4,12,6,14>(tmp2[2], tmp2[3]);
    matrix_T[3] = blend8<1,9,3,11,5,13,7,15>(tmp2[2], tmp2[3]);
    matrix_T[4] = blend8<0,8,2,10,4,12,6,14>(tmp2[4], tmp2[5]);
    matrix_T[5] = blend8<1,9,3,11,5,13,7,15>(tmp2[4], tmp2[5]);
    matrix_T[6] = blend8<0,8,2,10,4,12,6,14>(tmp2[6], tmp2[7]);
    matrix_T[7] = blend8<1,9,3,11,5,13,7,15>(tmp2[6], tmp2[7]); 

}

// do transpose of matrices in Vec4f by 4 or Vec8f by 8.
inline std::array<V,V::size()> _permuteV(const std::array<V,V::size()>& matrix){

    std::array<V,V::size()> matrix_T;
    
    _permuteV8(matrix.data(), matrix_T.data());

    return matrix_T;

}

inline std::array<V,M> Front_Option_3(const std::array<V,M>& x){

    std::array<V,M> x_T = _permuteV(x);

    std::array<V,M> w_T = T_ZIC(x_T);

    std::array<V,M> y_T = T_ICC(w_T);

    return y_T;

};


inline std::array<V,M> End_Option_3(const std::array<V,M>& x_T){

    std::array<V,M> w_T = T_ZIC(x_T);

    std::array<V,M> y_T = T_ICC(w_T);

    std::array<V,M> y = _permuteV(y_T);

    return y;

};

int main(){

    for (auto n=0; n<N; n++) {

        //

        Outputs[n] = End_Option_3(Front_Option_3(Inputs[n]));

        //

    }

    return 0;

}


