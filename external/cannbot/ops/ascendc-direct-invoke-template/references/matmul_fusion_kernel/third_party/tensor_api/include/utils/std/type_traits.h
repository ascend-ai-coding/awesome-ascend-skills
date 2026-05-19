/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Type traits for Ascend C device-side code.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_STD_TYPE_TRAITS_H
#define INCLUDE_TENSOR_API_UTILS_STD_TYPE_TRAITS_H

#include <type_traits>

namespace AscendC {
namespace Std {

using std::is_same;
using std::is_same_v;
using std::is_base_of;
using std::is_base_of_v;
using std::enable_if;
using std::enable_if_t;
using std::remove_cv;
using std::remove_reference;
using std::integral_constant;
using std::true_type;
using std::false_type;
using std::bool_constant;

template<bool v>
using bool_constant = std::bool_constant<v>;

template<typename T>
struct always_false : std::false_type {};

template<typename T>
inline constexpr bool always_false_v = always_false<T>::value;

} // namespace Std
} // namespace AscendC

#endif
