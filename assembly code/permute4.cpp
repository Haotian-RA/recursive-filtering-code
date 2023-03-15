#include <https://raw.githubusercontent.com/vectorclass/version2/master/vectorclass.h>
#include <array>
#include <vector>

using V = Vec4f;
using T = float;
constexpr int M1 = 4;
constexpr static int N = 1;

using VBlock = std::array<V, M1>;
alignas(64) std::array<VBlock, N> Inputs, Outputs;
alignas(64) std::array<V, M1> V_Inputs, V_Outputs;
V v_inputs, v_outputs;

template<typename V> inline void _permuteV4(const V matrix[4], V matrix_T[4]){

    // first round: swap lower left and upper right
    V tmp[4];

    tmp[0] = blend4<0,1,4,5>(matrix[0], matrix[2]);
    tmp[1] = blend4<0,1,4,5>(matrix[1], matrix[3]);
    tmp[2] = blend4<2,3,6,7>(matrix[0], matrix[2]);
    tmp[3] = blend4<2,3,6,7>(matrix[1], matrix[3]);

    // second/final round: swap lower left and upper right within each quadrant
    matrix_T[0] = blend4<0,4,2,6>(tmp[0], tmp[1]);
    matrix_T[1] = blend4<1,5,3,7>(tmp[0], tmp[1]);
    matrix_T[2] = blend4<0,4,2,6>(tmp[2], tmp[3]);
    matrix_T[3] = blend4<1,5,3,7>(tmp[2], tmp[3]);

}

// do transpose of matrices in Vec4f by 4 or Vec8f by 8.
inline std::array<V,V::size()> _permuteV(const std::array<V,V::size()>& matrix){

    std::array<V,V::size()> matrix_T;
    
    _permuteV4(matrix.data(), matrix_T.data());

    return matrix_T;

}


int main(){

    for (auto n=0; n<N; n++) {

        // 

        Outputs[n] = _permuteV(Inputs[n]);

        //

    }

    return 0;

}


