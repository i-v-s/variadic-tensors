#ifndef VT_UTILS_H
#define VT_UTILS_H
#include <tuple>

namespace vt {


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

/************* tupleSlice *************/

template<typename Tuple, std::size_t... I>
constexpr auto gather(const Tuple& a, std::index_sequence<I...>) noexcept
{
    return std::make_tuple(std::get<I>(a)...);
}

template <std::size_t offset, std::size_t ... indices>
std::index_sequence<(offset + indices)...> addOffset(std::index_sequence<indices...>)
{
    return {};
}
\
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

}

#endif // VT_UTILS_H
