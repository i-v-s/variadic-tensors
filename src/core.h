#ifndef VT_CORE_H
#define VT_CORE_H
#include <concepts>
#include <memory>
#include <atomic>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <numeric>
#include <type_traits>
#include <sstream>
#include <assert.h>

#include "./axis.h"
#include "./shape.h"
#include "./utils.h"
#include "./strides.h"

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
struct ItemTypeT<Pointer, typename std::enable_if_t<!std::is_pointer_v<Pointer>>> { using Type = typename Pointer::Item; };

template<typename Pointer>
using ItemType = typename ItemTypeT<Pointer>::Type;


/************* Slice *************/

template <Integer Begin, Integer End, Integer Step = std::integral_constant<int, 1>>
struct Slice
{
public:

    Slice(Begin begin, End end) :
        begin(begin), end(end)
    {}

    Slice(End end) :
        end(end)
    {}

    Slice() {}

    constexpr bool empty() const noexcept { return begin == end; }

    Begin begin;
    End end;
    Step step;
};

template<Integer End>
Slice(End) -> Slice<std::integral_constant<int, 0>, End, std::integral_constant<int, 1>>;

Slice() -> Slice<std::integral_constant<int, 0>, std::integral_constant<int, 0>, std::integral_constant<int, 1>>;

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
    using StridesType = typename SST::Strides;

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
    template<typename... Slices>
    Item &at(Slices... args)
    {
        auto slices = std::make_tuple(Slice(args)...);
        return *pointer;
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

    Tensor(Pointer pointer, typename SST::Pair && sst) :
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

    AllocatedTensor(const typename Parent::ShapeType &shape) :
        Parent(Pointer(Parent::bufferSize(shape)), shape)
    {}
};

template<typename Item, typename... Args>
using HeapTensor = AllocatedTensor<HeapBuffer, Item, Args...>;

}

#endif // VT_CORE_H
