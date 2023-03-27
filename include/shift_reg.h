#ifndef SHIFT_REG_H
#define SHIFT_REG_H 1


#include "vectorclass.h"
#include <utility>



template<typename V> class Shift{

        using T = decltype(std::declval<V>().extract(0));
        constexpr static int M = V::size(); 

    private:

        V _buffer{0};

    public:

        inline void shift(const T x){

            if constexpr (M == 4) _buffer = blend4<1,2,3,4>(_buffer, x); 
            if constexpr (M == 8) _buffer = blend8<1,2,3,4,5,6,7,8>(_buffer, x); 
            if constexpr (M == 16) _buffer = blend16<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>(_buffer, x);
             
        }; // scalar shift, T
            
        inline void shift(const V x){ _buffer = x; }; // vector shift, NT

        inline T operator[](const int idx){ return (idx < 0) ? _buffer[M+idx] : _buffer[idx]; }; // read 


};


#endif // header guard 
