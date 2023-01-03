#ifndef AXIS_H
#define AXIS_H
#include <sstream>
#include <limits>

namespace vt {

enum Sizing
{
    Dynamic = std::numeric_limits<int>::min(),
    Auto
};

struct Empty {};

std::ostream& operator<< (std::ostream& stream, const Empty& t) noexcept;

template <int ID, int SIZE = Dynamic, int STRIDE = Auto> struct Axis
{
    static_assert(SIZE != Auto, "Axis size cannot be Auto");
    constexpr static int id = ID;
    constexpr static int size = SIZE;
    constexpr static int stride = STRIDE;
    constexpr static bool dynamic = SIZE == Dynamic;
    constexpr static bool contiguous = STRIDE == Auto;
    using Size = std::conditional_t<dynamic, int, std::integral_constant<int, size>>;
    using Stride = std::conditional_t<
        STRIDE == Auto, Empty,
        typename std::conditional_t<STRIDE == Dynamic, int, std::integral_constant<int, stride>>
    >;

    friend std::ostream& operator<< (std::ostream& stream, const Axis& t) noexcept
    {
        stream << "A<" << ID << ":";
        if constexpr(dynamic) stream << "D:";
        else stream << SIZE << ":";
        if constexpr(contiguous) stream << "C";
        else if constexpr(STRIDE == Dynamic) stream << "D";
        else stream << STRIDE;
        return stream << ">";
    }
};

template<class T>
concept AxisLike =
        std::is_integral_v<decltype(T::size)>
        && std::is_integral_v<decltype(T::id)>
        && std::is_integral_v<decltype(T::stride)>
        && std::is_convertible_v<typename T::Size, int>;

template<class T>
concept ContiguousAxis =
        AxisLike<T> && T::contiguous;

template<AxisLike Axis> struct RemoveStrideT;

template<int ID, int SIZE, int STRIDE>
struct RemoveStrideT<Axis<ID, SIZE, STRIDE>> { using Type = Axis<ID, SIZE>; };

template<AxisLike Axis>
using RemoveStride = typename RemoveStrideT<Axis>::Type;

}
#endif // AXIS_H
