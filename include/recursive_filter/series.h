#ifndef SERIES_H
#define SERIES_H 1

#include <array>
#include <tuple>
#include "second_order_cores.h"

// form higher order recursive filter by cascading second order cores
template<typename... Types> class Series{

    private:
        
        // tuple of second order cores
        std::tuple<Types...> _t; 

        // cascaded function of option 1
        template<int i, typename U> inline U _proc_option1(const U& x) {
            if constexpr (i >= std::tuple_size<decltype(_t)>::value) {
                return x;          
            } else {
                U r = std::get<i>(_t).option1(x);
                return _proc_option1<i+1>(r);  
            };
        };

        // cascaded function of option 2
        template<int i, typename U> inline U _proc_option2(const U& x) {
            if constexpr (i >= std::tuple_size<decltype(_t)>::value) {
                return x;          
            } else {
                U r = std::get<i>(_t).option2(x);
                return _proc_option2<i+1>(r);  
            };
        };

        // cascaded function of option 3
        template<int i, typename U> inline U _proc_option3(const U& x) {
            if constexpr (i >= std::tuple_size<decltype(_t)>::value) {
                return x;          
            } else {
                U r = std::get<i>(_t).option3_middle(x);
                return _proc_option3<i+1>(r);  
            };
        };
        
    public:

        // default constructor
        Series(){};

        // Parameterized constructor, initialize a tuple of second order cores 
        Series(Types...types): _t(types...){};

        // pass one vector of samples into cascaded higher order filter of option 1
        template<typename U> inline U series_option1(const U& x) { 
            return _proc_option1<0>(x); 
        };

        // pass one matrix of samples into cascaded higher order filter of option 2
        template<typename U> inline U series_option2(const U& x) { 
            return _proc_option2<0>(x); 
        };

        // pass one matrix of samples into cascaded higher order filter of option 3
        template<typename U> inline U series_option3(const U& x) { 
            return _proc_option3<0>(x); 
        };

};


/* 
    a list of functions that define a more user-friendly class declaration of series that
    can be initialized by just giving the array of coefficients and initial conditions.
 */


template <class T> struct unwrap_refwrapper { 
    using type = T; 
};

template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> { 
    using type = T&; 
};

template <class T> using unwrap_decay_t = typename unwrap_refwrapper<typename std::decay<T>::type>::type;

template <class... Types> constexpr Series<unwrap_decay_t<Types>...> make_series(Types&&... args) {
    return Series<unwrap_decay_t<Types>...>(std::forward<Types>(args)...);
};

template<typename V, typename Array1, typename Array2, std::size_t... I>
auto make_series_from_coeffs(const Array1& coefs, const Array2& inits, std::index_sequence<I...>) {
    using Class = IirCoreOrderTwo<V>;
    return make_series(Class(coefs[I], inits[I])...); 
};

template<typename T, typename V, size_t N, typename indices = std::make_index_sequence<N>>
auto series_from_coeffs(const T (&coefs)[N][5], const T (&inits)[N][4]={0}) { 
    return make_series_from_coeffs<V>(coefs, inits, indices{});
};

#endif // header guard 










































