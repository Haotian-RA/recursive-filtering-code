#ifndef FILTER_H
#define FILTER_H 1


#include "series.h"
#include "permuteV.h"


// apply the cascaded second order iir filter to process a block size of samples.
template<typename T, int N> class Filter{ 
    
    using V = Vec8f;
    constexpr static int M = V::size();   

    private:

        using Series_t = decltype(series_from_coeffs<T,V>(std::declval<const T (&)[N][5]>(), std::declval<const T (&)[N][4]>())); 
        Series_t _S;
        
    public:


        Filter(const T (&coeffs)[N][5], const T (&inits)[N][4]): _S(series_from_coeffs<T,V>(coeffs, inits)){}; 

        // filter for samples in stream
        template<typename InputIt, typename OutputIt> inline OutputIt operator()(InputIt first, OutputIt last, OutputIt d_first){

            std::array<V,M> x, y;

            while (first <= last - M*M){

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  
               
                y = _S(x);

                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                first += M*M;
                d_first += M*M;

            }

            return d_first;

        };


        // filter for samples in stream
        template<typename InputIt, typename OutputIt> inline OutputIt Cascaded_option_1(InputIt first, OutputIt last, OutputIt d_first){

            V x, y;

            while (first <= last - M){

                x.load(&*first);  
               
                y = _S.Series_option_1(x);

                y.store(&*d_first);

                first += M;
                d_first += M;

            }

            return d_first;

        };


        // filter for samples in stream
        template<typename InputIt, typename OutputIt> inline OutputIt Cascaded_option_2(InputIt first, OutputIt last, OutputIt d_first){

            std::array<V,M> x, y;

            while (first <= last - M*M){

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  
               
                y = _S.Series_option_2(x);

                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                first += M*M;
                d_first += M*M;

            }

            return d_first;

        };


        // filter for samples in stream
        template<typename InputIt, typename OutputIt> inline OutputIt Cascaded_option_3(InputIt first, OutputIt last, OutputIt d_first){

            std::array<V,M> x, y, x_T, y_T;

            while (first <= last - M*M){

                for (auto n=0; n<M; n++) x[n].load(&*(first + n*M));  

                x_T = _permuteV(x);
                
                y_T = _S.Series_option_3(x_T);

                y = _permuteV(y_T);
               
                for (auto n=0; n<M; n++) y[n].store(&*(d_first + n*M));

                first += M*M;
                d_first += M*M;

            }

            return d_first;

        };

       
};


#endif // header guard 

