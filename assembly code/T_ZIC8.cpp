#include <https://raw.githubusercontent.com/vectorclass/version2/master/vectorclass.h>
#include <array>
#include <vector>

using V = Vec8f;
using T = float;
constexpr static int N = 1;
constexpr int M = 8;

using VBlock = std::array<V, V::size()>;
alignas(64) std::array<VBlock, N> Inputs, Outputs;
alignas(64) std::array<V, V::size()> V_Inputs, V_Outputs;
V v_inputs, v_outputs;

T _b1, _b2, _a1, _a2;

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

inline std::array<V,M> T_ZIC(const std::array<V,M>& x){

    std::array<V,M> v, w;

    V    xi2 = blend8<8,0,1,2,3,4,5,6>(x[M-2], _S[-2]);
    V    xi1 = blend8<8,0,1,2,3,4,5,6>(x[M-1], _S[-1]);

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

    // _S.shift(x[M-2][M-1]);
    // _S.shift(x[M-1][M-1]);

    return w; 

};

int main(){

    for (auto n=0; n<N; n++) {

        Outputs[n] = T_ZIC(Inputs[n]);

    }

    return 0;

}


