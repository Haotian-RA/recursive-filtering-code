#ifndef PERMUTEV_H
#define PERMUTEV_H 1

#include "vectorclass.h"
#include <array>


// do transpose of matrices in Vec4f by 4 or Vec8f by 8.
template<typename V> inline std::array<V,V::size()> _permuteV(const std::array<V,V::size()>& matrix){

    std::array<V,V::size()> matrix_T;
    
    if constexpr (V::size() == 4) _permuteV4(matrix.data(), matrix_T.data());

    if constexpr (V::size() == 8) _permuteV8(matrix.data(), matrix_T.data());

    return matrix_T;

}



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



template<typename V> inline void _permuteV8(const V matrix[8], V matrix_T[8]){

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


#endif
