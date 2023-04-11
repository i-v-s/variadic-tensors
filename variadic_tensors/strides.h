#ifndef VT_STRIDES_H
#define VT_STRIDES_H
#include "./utils.h"
#include "./axis.h"

namespace vt {


/************* ShapeStridesTuple *************/

template<AxisLike... Axes>
struct ShapeStridesTuple;

template<AxisLike Axis, AxisLike... Axes>
struct ShapeStridesTuple<Axis, Axes...>
{
    using Prev = ShapeStridesTuple<Axes...>;
    using PrevSize = typename Prev::Size;
    using Stride = std::conditional_t<Axis::contiguous, PrevSize, typename Axis::Stride>;
    using Size = Product<typename Axis::Size, Stride>;
    using Strides = PushFront<Stride, typename Prev::Strides>;
    using Shape = PushFront<typename Axis::Size, typename Prev::Shape>;
    using Pair = std::pair<Shape, Strides>;

    template<bool dynamic, Integer... Args>
    static Shape buildShape(Size size, Args... args) noexcept;

    template<Integer... Args>
    static Shape buildShape(Args... args) noexcept
    {
        using namespace std;
        if constexpr (Axis::dynamic)
                return apply([] (auto size, auto... args) {
                    return pushFront(size, Prev::buildShape(args...));
                }, forward_as_tuple(args...));
        else
            return pushFront(typename Axis::Size{}, Prev::buildShape(args...));
    }

    template<Integer... Args>
    static Strides buildStrides(const Shape &shape, Args... args) noexcept
    {
        using namespace std;
        if constexpr (integral<Stride>) {
            if constexpr(Axis::contiguous) {
                auto strides = Prev::buildStrides(tupleTail(shape), args...);
                int prevStride = tuple_size_v<decltype(strides)> ? get<0>(strides) : 1;
                int prevSize = tuple_size_v<decltype(strides)> ? get<1>(shape) : 1;
                return pushFront(prevStride * prevSize, strides);
            } else
                return apply([&shape] (auto stride, auto... args) {
                    return pushFront(stride, Prev::buildStrides(tupleTail(shape), args...));
                }, forward_as_tuple(args...));
        } else
            return pushFront(Stride{}, Prev::buildStrides(tupleTail(shape), args...));
    }

    template<Integer... Args>
    static std::pair<Shape, Strides> build(Args... args) noexcept
    {
        using namespace std;
        auto at = make_tuple(args...);
        constexpr int sizeCount = ((Axes::dynamic ? 1 : 0) + ...) + (Axis::dynamic ? 1 : 0);
        static_assert(sizeCount <= sizeof...(args), "Not enough arguments provided");
        Shape shape = apply([] (auto... args) {
            return buildShape(args...);
        }, tupleSlice<0, sizeCount>(at));
        Strides strides = apply([&shape] (auto... args) {
            return buildStrides(shape, args...);
        }, tupleSlice<sizeCount>(at));
        return std::make_pair(shape, strides);
    }
};

template<> struct ShapeStridesTuple<>
{
    using Size = std::integral_constant<int, 1>;
    using Strides = std::tuple<>;
    using Shape = std::tuple<>;

    static constexpr Shape buildShape() noexcept { return Shape{}; }
    static constexpr Strides buildStrides(const Shape&) noexcept { return Strides{}; }
};

template<AxisLike... Axes, std::integral... Args>
auto buildShapeStrides(Args... args)
{
    return ShapeStridesTuple<Axes...>::build(args...);
}


/************* commonStrides *************/

template<typename Value>
auto commonStridesReductor(const Value &value, Bool auto contiguous, Integer auto size, Integer auto stride1, Integer auto stride2) noexcept
{
    using namespace std;
    auto &[count, result] = value;
    if constexpr(contiguous)
        return make_tuple(product(count, size), result);
    else
        return make_tuple(size, pushFront(make_tuple(count, stride1, stride2), result)); // stride ???
}

template<size_t itemSize, typename Contiguous, typename Shape, typename Strides1, typename Strides2>
auto commonStrides(const Contiguous &contiguous, const Shape &shape, const Strides1 &strides1, const Strides2 &strides2)
{
    using namespace std;
    return apply([] (const auto& ... args) {
        return reduce([] (auto... args) {
            return commonStridesReductor(args...);
        }, make_tuple(IntConst<itemSize>(), make_tuple()), args...);
    }, zip<true>(contiguous, shape, strides1, strides2));
}

}

#endif // VT_STRIDES_H
