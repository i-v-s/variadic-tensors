#ifndef VT_STRIDES_H
#define VT_STRIDES_H
#include "./utils.h"
#include "./axis.h"

namespace vt {

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
            return pushFront(Size{}, Prev::buildShape(args...));
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


}

#endif // VT_STRIDES_H
