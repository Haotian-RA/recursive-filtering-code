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

Shift<V> _S;

inline std::array<V,M> T_ICC(const std::array<V,M>& w){

    std::array<V,M> y;

    // step 1: initialization
    y[M-2] = mul_add(_rd0_tl, _S[-2], w[M-2]);
    y[M-2] = mul_add(_rd0_bl, _S[-1], y[M-2]);
    y[M-1] = mul_add(_rd0_tr, _S[-2], w[M-1]);
    y[M-1] = mul_add(_rd0_br, _S[-1], y[M-1]);

    // step 2: first recursion
    V tmp2 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-2]);
    V tmp1 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-1]);

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
    V yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
    V yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);

    for (auto i=0; i<M-2; i++){
        
        y[i] = mul_add(yi2, _h2[i], w[i]);
        y[i] = mul_add(yi1, _h1[i], y[i]);

    };

    _S.shift(y[M-2][M-1]);
    _S.shift(y[M-1][M-1]); 
    
    return y; 

};

int main(){

    for (auto n=0; n<N; n++) {

        //

        Outputs[n] = T_ICC(Inputs[n]);

        //

    }

    return 0;

}


