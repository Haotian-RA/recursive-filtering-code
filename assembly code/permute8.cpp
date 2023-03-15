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


int main(){

    for (auto n=0; n<N; n++) {

        // 

        Outputs[n] = _permuteV(Inputs[n]);

        //

    }

    return 0;

}


