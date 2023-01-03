#ifndef VT_UTILS_H
#define VT_UTILS_H
#include <tuple>
#include <concepts>
#include <functional>

namespace vt {

/************* Types *************/


template<typename T>
concept Integer = std::convertible_to<T, int>;

template<typename T>
concept Bool = std::convertible_to<T, bool>;

template<int value>
using IntConst = std::integral_constant<int, value>;

template<bool value>
using BoolConst = std::integral_constant<bool, value>;

template<typename T>
concept IntConstLike =
        Integer<T>
        && !requires(T &x) { x = int(0); };

/************* Operate *************/

template<typename O, Integer A, Integer B> struct OperateT {
    using Result = int;
};

template<typename O, int a, int b>
struct OperateT<O, std::integral_constant<int, a>, std::integral_constant<int, b>> {
    using Result = std::integral_constant<int, O()(a, b)>;
};

template<typename O, Integer A, Integer B>
using Operate = typename OperateT<O, A, B>::Result;

template<typename O, Integer A, Integer B>
Operate<O, A, B> operate(O && o, A a, B b) noexcept
{
    using Result = Operate<O, A, B>;
    if constexpr(IntConstLike<Result>)
        return Result();
    else
        return o(a, b);
}

template<Integer A, Integer B>
using Product = Operate<std::multiplies<int>, A, B>;

auto product(Integer auto a, Integer auto b) noexcept
{
    return operate(std::multiplies<int>(), a, b);
}

auto add(Integer auto a, Integer auto b) noexcept
{
    return operate(std::plus<int>(), a, b);
}

template<int b = 1>
auto add(Integer auto a) noexcept
{
    return operate(std::plus<int>(), a, std::integral_constant<int, b>());
}

auto sub(Integer auto a, Integer auto b) noexcept
{
    return operate(std::minus<int>(), a, b);
}

auto div(Integer auto a, Integer auto b) noexcept
{
    return operate(std::divides<int>(), a, b);
}



/************* TupleCat *************/

template<typename... Tuples> struct TupleCatT;

    template<typename... Args1, typename... Args2, typename... Tuples>
    struct TupleCatT<std::tuple<Args1...>, std::tuple<Args2...>, Tuples...>
    {
        using Type = typename TupleCatT<std::tuple<Args1..., Args2...>, Tuples...>::Type;
    };

    template<typename Tuple> struct TupleCatT<Tuple> { using Type = Tuple; };
    template<> struct TupleCatT<> { using Type = std::tuple<>; };

    template<typename... Tuples>
    using TupleCat = typename TupleCatT<Tuples...>::Type;

template<typename Tuple, typename Item>
using PushBack = TupleCat<Tuple, std::tuple<Item>>;

template<typename Item, typename Tuple>
using PushFront = TupleCat<std::tuple<Item>, Tuple>;

template<typename Item, typename Tuple>
auto pushFront(Item && item, Tuple && tuple) noexcept {
    using namespace std;
    return tuple_cat(make_tuple(forward<Item>(item)), forward<Tuple>(tuple));
}

template<typename Tuple, typename Item>
auto pushBack(Tuple && tuple, Item && item) noexcept {
    using namespace std;
    return tuple_cat(forward<Tuple>(tuple), make_tuple(forward<Item>(item)));
}


/************* gather *************/

/// Gather from tuple by sequence of indices
template<typename Tuple, std::size_t... I>
constexpr auto gather(const Tuple& a, std::index_sequence<I...>) noexcept
{
    return std::make_tuple(std::get<I>(a)...);
}

/// Gather from many tuples by one index
template<std::size_t I, typename... Tuples>
constexpr auto gather(const Tuples& ... a) noexcept
{
    return std::make_tuple(std::get<I>(a)...);
}

/// Gather from many tuples by sequence of indices
template<std::size_t... I, typename... Tuples>
constexpr auto gather(std::index_sequence<I...>, const Tuples& ... a) noexcept
{
    using namespace std;
    return make_tuple(gather<I>(a...)...);
}


/************* tupleSlice *************/

template <std::size_t offset, std::size_t ... indices>
std::index_sequence<(offset + indices)...> addOffset(std::index_sequence<indices...>) { return {}; }

template <std::size_t offset, std::size_t ... indices>
std::index_sequence<(offset - indices)...> subOffset(std::index_sequence<indices...>) { return {}; }

template<int b, int e = -1, typename... Args>
constexpr auto tupleSlice(const std::tuple<Args...> &t) noexcept
{
    constexpr int end = (e == -1) ? sizeof...(Args) : e;
    if constexpr(end == b)
        return std::make_tuple();
    static_assert(end >= b, "Length must be positve");
    static_assert(b >= 0 && end <= sizeof...(Args), "Indexes must be in range");
    return gather(t, addOffset<b>(std::make_index_sequence<end - b>{}));
}

template<typename... Args>
constexpr auto tupleTail(const std::tuple<Args...> &t) noexcept
{
    return tupleSlice<1>(t);
}

template<typename... Args>
constexpr auto tupleToArray(const std::tuple<Args...> &t)
{
    return std::apply([] (auto... args) { return std::array<int, sizeof...(args)>{static_cast<int>(args)...}; }, t);
}


/************* zip *************/

template<bool reverse = false, typename Tuple, typename... Tuples>
auto zip(const Tuple &arg, const Tuples&... args)
{
    using namespace std;
    constexpr auto size = tuple_size_v<Tuple>;
    static_assert(((size == tuple_size_v<Tuples>) && ...), "Tuples must be the same size");
    if constexpr(reverse)
        return gather(subOffset<size - 1>(make_index_sequence<size>{}), arg, args...);
    else
        return gather(make_index_sequence<size>{}, arg, args...);
}

template<bool reverse = false, typename Tuple>
auto applyZip(const Tuple &arg)
{
    using namespace std;
    return apply([] (auto... args) {
        return zip<reverse>(args...);
    }, arg);
}


/************* Apply *************/

template<template<typename...> typename Target, typename Source>
struct ApplyT;

template<template<typename...> typename Target, typename... Args>
struct ApplyT<Target, std::tuple<Args...>> { using Type = Target<Args...>; };

template<template<typename...> typename Target, typename Source>
using Apply = typename ApplyT<Target, Source>::Type;


/************* tupleFill *************/

template<std::size_t size, typename Item, typename Tuple = std::tuple<>>
auto tupleFill(const Item &item, const Tuple &tuple = Tuple())
{
    using namespace std;
    constexpr auto tupleSize = tuple_size_v<Tuple>;
    static_assert(tupleSize <= size, "Tuple size is larger then requested");
    if constexpr(tupleSize == size)
        return tuple;
    else
        return pushBack(tupleFill<size - 1>(item, tuple), item);
}


/************* true seq *************/

template<typename a, typename b> struct SeqCatT;

template<std::size_t... a, std::size_t... b>
struct SeqCatT<std::index_sequence<a...>, std::index_sequence<b...>>
{
    using Type = std::index_sequence<a..., b...>;
};

template<typename a, typename b>
using SeqCat = typename SeqCatT<a, b>::Type;

template<std::size_t i, bool... bs>
struct TrueSeqT;

template<std::size_t i, bool b, bool... bs>
struct TrueSeqT<i, b, bs...>
{
    using NextType = typename TrueSeqT<i + 1, bs...>::Type;
    using Type = std::conditional_t<b, SeqCat<std::index_sequence<i>, NextType>, NextType>;
};

template<std::size_t i>
struct TrueSeqT<i> {using Type = std::index_sequence<>;};

template<bool... bs>
using TrueSeq = typename TrueSeqT<0, bs...>::Type;


/************* TupleReduce *************/

template<typename Fn, typename Value>
auto reduce(Fn fn, Value value) noexcept
{
    return value;
}

template<typename Fn, typename Value, typename Arg, typename... Args>
auto reduce(Fn &&fn, const Value value, Arg arg, Args... args) noexcept
{
    return reduce(
        fn,
        std::apply([&] (auto... args) { return fn(value, args...); }, arg),
        args...);
}


/************* countIf *************/

template<typename Fn, typename Tuple>
constexpr int countIf(Fn && fn, const Tuple &tp) noexcept
{
    return std::apply([&fn] (const auto&... args) {
        return ((fn(args) ? 1 : 0) + ...);
    }, tp);
}

}

#endif // VT_UTILS_H
