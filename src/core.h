#ifndef COMMON_TENSOR_H
#define COMMON_TENSOR_H
#include <concepts>
#include <memory>
#include <atomic>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <numeric>
#include <type_traits>
#include <limits>
#include <sstream>
#include <assert.h>

#include "./axis.h"
#include "./shape.h"
#include "./utils.h"

template<typename Arg, typename... Args>
std::ostream& operator<<(std::ostream &stream, const std::tuple<Arg, Args...> &t) noexcept
{
    stream << "[";
    std::apply([&stream] (Arg arg, Args... args) {
        stream << arg;
        ((stream << ", " << args), ...);
    }, t);
    stream << "]";
    return stream;
}

namespace vt {


/************* Strides *************/

template<typename T>
concept Integer = std::convertible_to<T, int>;

template<Integer A, Integer B> struct ProductT {
    using Result = int;
};

template<int a, int b>
struct ProductT<std::integral_constant<int, a>, std::integral_constant<int, b>> {
    using Result = std::integral_constant<int, a * b>;
};

template<Integer A, Integer B>
using Product = ProductT<A, B>::Result;

template<AxisLike... Axes>
struct ShapeStridesTuple;

template<AxisLike Axis, AxisLike... Axes>
struct ShapeStridesTuple<Axis, Axes...>
{
    using Prev = ShapeStridesTuple<Axes...>;
    using PrevSize = Prev::Size;
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


/************* Buffers *************/

template<class T>
concept BufferLike =
        std::is_constructible_v<T, size_t>
        && !std::copyable<T>;

template<typename Derived>
class Buffer
{
public:
    Buffer() = delete;
    Buffer(size_t size) : memory(Derived::malloc(size), &Derived::dealloc) {}

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;

    void *get() noexcept { return memory.get(); }
    const void *get() const noexcept { return memory.get(); }
private:
    std::unique_ptr<void, void (*)(void *)> memory;
};

class HeapBuffer
{
public:
    HeapBuffer(size_t size);

    HeapBuffer(const HeapBuffer &) = delete;
    HeapBuffer(HeapBuffer &&) = delete;
    HeapBuffer &operator=(const HeapBuffer &) = delete;
    HeapBuffer &operator=(HeapBuffer &&) = delete;

    void *get() noexcept;
    const void *get() const noexcept;
private:
    std::unique_ptr<uint8_t[]> memory;
};


/************* Pointers *************/

template<BufferLike Buffer, typename Item, bool offset = true> class SharedPointer;

template<BufferLike Buffer_, typename Item_>
class SharedPointer<Buffer_, Item_, false>
{
public:
    using Item = Item_;
    using Buffer = Buffer_;
    operator Item*() noexcept
    {
        return static_cast<Item *>(value->get());
    }
    operator Item*() const noexcept
    {
        return static_cast<Item *>(value->get());
    }

    explicit SharedPointer(size_t size) :
        value(new Buffer(size))
    {}
    SharedPointer(SharedPointer && other) :
        value(std::move(other.value))
    {}
    SharedPointer(const SharedPointer & other) :
        value(other.value)
    {}
protected:
    std::shared_ptr<Buffer> value;
};

/************* ItemType *************/

template<typename Pointer, class Enable = void> struct ItemTypeT;

template<typename Pointer>
struct ItemTypeT<Pointer, typename std::enable_if_t<std::is_pointer_v<Pointer>>> { using Type = std::remove_pointer_t<Pointer>; };

template<typename Pointer>
struct ItemTypeT<Pointer, typename std::enable_if_t<!std::is_pointer_v<Pointer>>> { using Type = Pointer::Item; };

template<typename Pointer>
using ItemType = ItemTypeT<Pointer>::Type;


/************* Tensors *************/

//template<class Tensor>
//concept TensorLike =

template<BufferLike Buffer, typename Item, typename... TensorArgs> class AllocatedTensor;

template<typename Pointer, typename... Axes>
class Tensor
{
public:
    using SST = ShapeStridesTuple<Axes...>;
    using Item = ItemType<Pointer>;
    using ShapeType = Shape<typename SST::Shape>;
    using StridesType = SST::Strides;

    Tensor(Pointer pointer, const ShapeType &shape) :
        shape(shape),
        strides(SST::buildStrides(shape)),
        pointer(pointer)
    {}

    template<std::integral... Args>
    Tensor(Item *pointer, Args... args) : Tensor(pointer, SST::build(args...))
    {}

    auto operator[](uint idx) noexcept
    {
        return *this;
    }
    template<std::integral... Args>
    Item &at(Args... args)
    {

    }

    template<typename Other>
    operator Tensor<Other, Axes...>() const
    {
        Other result(shape);
        return result;
    }

    template<BufferLike Buffer>
    auto to() const
    {
        AllocatedTensor<Buffer, Item, RemoveStride<Axes>...> result(shape);

        return result;
    }

    const ShapeType shape;
    const StridesType strides;
protected:

    Tensor(Pointer pointer, SST::Pair && sst) :
        shape(sst.first),
        strides(sst.second),
        pointer(pointer)
    {}

    static size_t bufferSize(const ShapeType &shape) noexcept
    {
        static_assert((Axes::contiguous && ...), "Unable to calculate size for non-contiguous tensor");
        return sizeof(Item) * shape.total();
    }

    template<std::integral... Args>
    static size_t bufferSize(Args... args) noexcept
    {
        return bufferSize(ShapeType(tupleSlice<0, sizeof...(Axes)>(SST::build(args...).first)));
    }

    Pointer pointer;
};

template<typename Item, typename... Args>
using PassiveTensor = Tensor<Item *, Args...>;

template<BufferLike Buffer, typename Item, typename... TensorArgs>
class AllocatedTensor: public Tensor<SharedPointer<Buffer, Item, false>, TensorArgs...>
{
public:
    using Pointer = SharedPointer<Buffer, Item, false>;
    using Parent = Tensor<Pointer, TensorArgs...>;

    template<std::integral... Args>
    AllocatedTensor(Args... args) :
        Parent(Pointer(Parent::bufferSize(args...)), args...)
    {}

    AllocatedTensor(const Parent::ShapeType &shape) :
        Parent(Pointer(Parent::bufferSize(shape)), shape)
    {}
};

template<typename Item, typename... Args>
using HeapTensor = AllocatedTensor<HeapBuffer, Item, Args...>;

}

#endif // COMMON_TENSOR_H
