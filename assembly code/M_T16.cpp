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


int main(){

    for (auto n=0; n<N; n++) {

        Outputs[n] = _permuteV(Inputs[n]);

    }

    return 0;

}

