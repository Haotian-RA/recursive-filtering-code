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
V _f1, _f2;
std::array<V,M> _H;

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

inline V Option_1(const V x){

    V y{0};

    for (auto n=0; n<M; n++) y = mul_add(_H[n], x[n], y);

    y = mul_add(_f2, _S1[-2], y);
    y = mul_add(_f1, _S1[-1], y);

    y = mul_add(_h2, _S2[-2], y);
    y = mul_add(_h1, _S2[-1], y);

    _S1.shift(x);
    _S2.shift(y);

    return y;

};


int main(){

    for (auto n=0; n<N; n++) {

        //

        v_outputs = Option_1(v_inputs);

        //

    }

    return 0;

}


