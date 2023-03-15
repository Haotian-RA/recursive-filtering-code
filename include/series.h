#ifndef SERIES_H
#define SERIES_H 1


#include <tuple>
#include "iir_cores.h"
#include <array>


template<typename... Types> class Series{

    private:
        
        std::tuple<Types...> _t; 

        template<int i, typename U> inline U _proc(const U& x){

            if constexpr (i >= std::tuple_size<decltype(_t)>::value){

                return x;          
            } else{

                U r = std::get<i>(_t).Option_2(x);
                return _proc<i+1>(r);  

            };

        };

        template<int i, typename U> inline U _proc_option_1(const U& x){

            if constexpr (i >= std::tuple_size<decltype(_t)>::value){

                return x;          
            } else{

                U r = std::get<i>(_t).Option_1(x);
                return _proc_option_1<i+1>(r);  

            };

        };

        template<int i, typename U> inline U _proc_option_2(const U& x){

            if constexpr (i >= std::tuple_size<decltype(_t)>::value){

                return x;          
            } else{

                U r = std::get<i>(_t).Option_2(x);
                return _proc_option_2<i+1>(r);  

            };

        };

        template<int i, typename U> inline U _proc_option_3(const U& x){

            if constexpr (i >= std::tuple_size<decltype(_t)>::value){

                return x;          
            } else{

                U r = std::get<i>(_t).Mid_Option_3(x);
                return _proc_option_3<i+1>(r);  

            };

        };

        
    public:

        Series(Types...types): _t(types...){};


        template<typename U> inline U operator()(const U& x){ return _proc<0>(x); };

        template<typename U> inline U Series_option_1(const U& x){ return _proc_option_1<0>(x); };

        template<typename U> inline U Series_option_2(const U& x){ return _proc_option_2<0>(x); };

        template<typename U> inline U Series_option_3(const U& x){ return _proc_option_3<0>(x); };


};


// a more user-friendly class declaration by giving the coefficients and pre states.
template <class T> struct unwrap_refwrapper{ using type = T; };


template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>>{ using type = T&; };


template <class T> using unwrap_decay_t = typename unwrap_refwrapper<typename std::decay<T>::type>::type;


template <class... Types> constexpr Series<unwrap_decay_t<Types>...> make_series(Types&&... args){

  return Series<unwrap_decay_t<Types>...>(std::forward<Types>(args)...);

};


template<typename V, typename Array1, typename Array2, std::size_t... I>
auto make_series_from_coeffs(const Array1& coefs, const Array2& inits, std::index_sequence<I...>){

    using Class = IirCoreOrderTwo<V>;
    return make_series(Class(coefs[I], inits[I])...); 

};


template<typename T, typename V, size_t N, typename indices = std::make_index_sequence<N>>
auto series_from_coeffs(const T (&coefs)[N][5], const T (&inits)[N][4]={0}){ 

    return make_series_from_coeffs<V>(coefs, inits, indices{});

};



#endif // header guard 










































