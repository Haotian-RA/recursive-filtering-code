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

V _f1, _f2;
std::array<V,M> _H;

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

Shift<V> _S;

inline V NT_ZIC(const V x){

    V w{0};

    for (auto n=0; n<M; n++) w = mul_add(_H[n], x[n], w);

    w = mul_add(_f2, _S[-2], w);
    w = mul_add(_f1, _S[-1], w);

    _S.shift(x);

    return w; 

};

int main(){

    for (auto n=0; n<N; n++) {

        v_outputs = NT_ZIC(v_inputs);

    }

    return 0;

}

