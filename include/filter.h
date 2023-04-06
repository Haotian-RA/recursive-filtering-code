#ifndef FILTER_H
#define FILTER_H 1

#include "series.h"
#include "permuteV.h"

// real function to user: use the cascaded second order filter to process a trunk of data.
template<typename T, int N> class Filter{ 
    
    // select the vector length and type based on the requested instruction set and the type T
    #if INSTRSET >= 9  // AVX512
        using _V = typename std::conditional<std::is_same<T, float>::value, Vec16f, Vec8d>::type;
    #elif INSTRSET >= 7  // AVX2
        using _V = typename std::conditional<std::is_same<T, float>::value, Vec8f, Vec4d>::type;
    #else // SSE
        using _V = typename std::conditional<std::is_same<T, float>::value, Vec4f, Vec2d>::type;
    #endif

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    private:

        // define state of series from array of coefficients and initial conditions. 
        using Series_t = decltype(series_from_coeffs<T,V>(std::declval<const T (&)[N][5]>(), std::declval<const T (&)[N][4]>())); 
        Series_t _S;
        
    public:

        // default constructor
        Filter(){};

        // Parameterized constructor, initialize higher order filter by array of coefficients and pre-conditions
        Filter(const T (&coeffs)[N][5], const T (&inits)[N][4]): _S(series_from_coeffs<T,V>(coeffs, inits)){}; 


        /* 
        
            Higher order recursive filter that accepts a trunk of data, including
            cascaded_option1: block filtering
            cascaded_option2: mixed block and multi-block filtering 
            cascaded_option3: multi-block filtering 
            operator: multi-block filtering, the same as cascaded_option3.

         */


        // higher order filter of cascaded option 1, block filtering: filtering a vector of data.
        template<typename InputIt, typename OutputIt> inline OutputIt cascaded_option1(InputIt first, OutputIt last, OutputIt d_first) {
            V x, y;

            while (first <= last - M) {

                x.load(&*first);  
               
                y = _S.series_option1(x);

                y.store(&*d_first);

                // iterator += size of one vector
                first += M;
                d_first += M;
            }

            // return the iterator of the last data
            return d_first;
        };

        // higher order filter of cascaded option 2, mixed filtering: filtering a matrix of data.
        template<typename InputIt, typename OutputIt> inline OutputIt cascaded_option2(InputIt first, OutputIt last, OutputIt d_first) {
            std::array<V,M> x, y;

            while (first <= last - M*M) {

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  
               
                y = _S.series_option2(x);

                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                // iterator += size of one matrix
                first += M*M;
                d_first += M*M;
            }

            return d_first;
        };

        // higher order filter of cascaded option 3, multi-block filtering: filtering a matrix of data.
        template<typename InputIt, typename OutputIt> inline OutputIt cascaded_option3(InputIt first, OutputIt last, OutputIt d_first) {
            std::array<V,M> x, y, x_T, y_T;

            while (first <= last - M*M){

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  

                x_T = _permuteV(x);
                y_T = _S.series_option3(x_T);
                y = _permuteV(y_T);
               
                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                // iterator += size of one matrix
                first += M*M;
                d_first += M*M;

            }

            return d_first;
        };

        // operator, higher order filter of cascaded option 3.
        template<typename InputIt, typename OutputIt> inline OutputIt operator()(InputIt first, OutputIt last, OutputIt d_first) {
            std::array<V,M> x, y, x_T, y_T;

            while (first <= last - M*M){

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  

                x_T = _permuteV(x);
                y_T = _S.series_option3(x_T);
                y = _permuteV(y_T);
               
                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                // iterator += size of one matrix
                first += M*M;
                d_first += M*M;

            }

            return d_first;
        };

};

#endif // header guard 

