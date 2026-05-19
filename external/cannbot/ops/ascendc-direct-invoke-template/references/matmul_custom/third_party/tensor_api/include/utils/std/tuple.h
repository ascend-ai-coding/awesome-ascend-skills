/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Custom tuple implementation for Ascend C device-side code.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_STD_TUPLE_H
#define INCLUDE_TENSOR_API_UTILS_STD_TUPLE_H

#include <cstdint>
#include <cstddef>

namespace AscendC {
namespace Std {

// Forward declarations
template<typename... Types>
struct tuple;

// tuple_size
template<typename T>
struct tuple_size;

template<typename... Types>
struct tuple_size<tuple<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

template<typename T>
inline constexpr size_t tuple_size_v = tuple_size<T>::value;

// tuple_element
template<size_t I, typename T>
struct tuple_element;

template<typename Head, typename... Tail>
struct tuple_element<0, tuple<Head, Tail...>> {
    using type = Head;
};

template<size_t I, typename Head, typename... Tail>
struct tuple_element<I, tuple<Head, Tail...>> {
    using type = typename tuple_element<I-1, tuple<Tail...>>::type;
};

template<size_t I, typename T>
using tuple_element_t = typename tuple_element<I, T>::type;

// Integer sequence
template<typename T, T... Ints>
struct integer_sequence {
    using value_type = T;
    static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

// make_index_sequence
template<size_t N>
struct make_index_sequence_impl;

template<>
struct make_index_sequence_impl<0> {
    using type = index_sequence<>;
};

template<>
struct make_index_sequence_impl<1> {
    using type = index_sequence<0>;
};

template<>
struct make_index_sequence_impl<2> {
    using type = index_sequence<0, 1>;
};

template<>
struct make_index_sequence_impl<3> {
    using type = index_sequence<0, 1, 2>;
};

template<>
struct make_index_sequence_impl<4> {
    using type = index_sequence<0, 1, 2, 3>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

// Integral constant
template<typename T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

template<size_t v>
using Int = integral_constant<size_t, v>;

// Bool constants
template<bool v>
using bool_constant = integral_constant<bool, v>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

// remove_cvref
template<typename T>
struct remove_cvref {
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// is_one_of
template<typename T, typename... Args>
struct is_one_of;

template<typename T>
struct is_one_of<T> : false_type {};

template<typename T, typename First, typename... Rest>
struct is_one_of<T, First, Rest...>
    : bool_constant<std::is_same<T, First>::value || is_one_of<T, Rest...>::value> {};

template<typename T, typename... Args>
inline constexpr bool is_one_of_v = is_one_of<T, Args...>::value;

// ignore_t
struct ignore_t {};

// Simple tuple implementation (minimal, for compile-time use)
template<>
struct tuple<> {};

template<typename Head>
struct tuple<Head> {
    Head head;
};

template<typename Head, typename... Tail>
struct tuple<Head, Tail...> {
    Head head;
    tuple<Tail...> tail;
};

// get
template<size_t I, typename... Types>
constexpr auto get(const tuple<Types...>& t);

template<typename Head, typename... Tail>
constexpr Head get(const tuple<Head, Tail...>& t) { return t.head; }

template<size_t I, typename Head, typename... Tail>
constexpr auto get(const tuple<Head, Tail...>& t) {
    return get<I-1>(t.tail);
}

// Type traits passthrough from std
using std::is_same;
using std::is_base_of;
using std::enable_if;
using std::enable_if_t;

} // namespace Std
} // namespace AscendC

#endif
