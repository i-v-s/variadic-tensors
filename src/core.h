#ifndef VT_CORE_H
#define VT_CORE_H
#include <concepts>
#include <memory>
#include <atomic>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <sstream>
#include <assert.h>

#include "./axis.h"
#include "./shape.h"
#include "./utils.h"
#include "./strides.h"
#include "./slice.h"

namespace vt {

inline std::ostream& operator<<(std::ostream &stream, std::tuple<>) noexcept
{
    return stream << "[]";
}

template<typename T, T v>
inline std::ostream& operator<<(std::ostream &stream, std::integral_constant<T, v>) noexcept
{
    return stream << v << "c";
}

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


/************* Buffers *************/

enum class Device
{
    Host,
    Cuda,
    OpenCL
};

class PassiveBuffer {
public:
    constexpr static Device device = Device::Host;
};

template<class T>
concept BufferLike =
        (std::is_constructible_v<T, size_t> && !std::copyable<T>) || std::is_same_v<T, PassiveBuffer>;

template<class T>
concept HostBufferLike =
        BufferLike<T> && T::device == Device::Host;

template<class T>
concept CudaBufferLike =
        BufferLike<T> && T::device == Device::Cuda;

template<typename Derived>
class Buffer
{
public:
    constexpr static Device device = Device::Host;
    Buffer() = delete;
    Buffer(size_t size) : memory(Derived::malloc(size), &Derived::dealloc) {}

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;
    operator PassiveBuffer(){ return {}; }

    void *get() noexcept { return memory.get(); }
    const void *get() const noexcept { return memory.get(); }
private:
    std::unique_ptr<void, void (*)(void *)> memory;
};

class HeapBuffer
{
public:
    constexpr static Device device = Device::Host;
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

/************* GetItem, GetBuffer *************/

template<typename Pointer, class Enable = void> struct GetItemTypeT;

    template<typename Pointer>
    struct GetItemTypeT<Pointer, typename std::enable_if_t<std::is_pointer_v<Pointer>>> { using Type = std::remove_pointer_t<Pointer>; };

    template<typename Pointer>
    struct GetItemTypeT<Pointer, typename std::enable_if_t<!std::is_pointer_v<Pointer>>> { using Type = typename Pointer::Item; };

    template<typename Pointer>
    using GetItem = typename GetItemTypeT<Pointer>::Type;

template<typename Pointer, class Enable = void> struct GetBufferTypeT;

    template<typename Pointer>
    struct GetBufferTypeT<Pointer, typename std::enable_if_t<std::is_pointer_v<Pointer>>> { using Type = PassiveBuffer; };

    template<typename Pointer>
    struct GetBufferTypeT<Pointer, typename std::enable_if_t<!std::is_pointer_v<Pointer>>> { using Type = typename Pointer::Buffer; };

    template<typename Pointer>
    using GetBuffer = typename GetBufferTypeT<Pointer>::Type;


/************* Actions *************/

template<BufferLike SrcBuffer, BufferLike DstBuffer> struct Copy;

struct HostCopy
{
    static void copy(const void* src, void *dst, size_t size);
    static void copy(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides);
};

template<HostBufferLike Src, HostBufferLike Dst>
struct Copy<Src, Dst> : HostCopy {};


/************* Tensors *************/

//template<class Tensor>
//concept TensorLike =

template<BufferLike Buffer, typename Item, typename... TensorArgs> class AllocatedTensor;

template<typename Pointer_> class ConstReference
{
public:
    using Pointer = Pointer_;
    using Item = GetItem<Pointer>;
    ConstReference() = delete;
    ConstReference(Pointer pointer) : pointer(pointer) {}
    ConstReference(ConstReference &&) = default;
    Item operator() () const noexcept { return *pointer; }
protected:
    Pointer pointer;
};

template<typename Pointer>
class Reference: public ConstReference<Pointer>
{
public:
    using ConstReference<Pointer>::ConstReference;
    Reference(Pointer pointer) : ConstReference<Pointer>(pointer) {}
    Reference &operator=(const GetItem<Pointer> &item) noexcept
    {
        *ConstReference<Pointer>::pointer = item;
        return *this;
    }
};

template<typename Pointer_, AxisLike... Axes>
class Tensor
{
public:
    using Pointer = Pointer_;
    using SST = ShapeStridesTuple<Axes...>;
    using Ids = std::integer_sequence<int, Axes::id...>;
    using Item = GetItem<Pointer>;
    using ShapeType = Shape<typename SST::Shape>;
    using StridesType = typename SST::Strides;

    Tensor(Pointer pointer, const ShapeType &shape) :
        shape(shape),
        strides(SST::buildStrides(shape)),
        pointer(pointer)
    {}

    Tensor(Pointer pointer, const ShapeType &shape, const StridesType &strides) :
        shape(shape),
        strides(strides),
        pointer(pointer)
    {}

    template<std::integral... Args>
    Tensor(Pointer pointer, Args... args) : Tensor(pointer, SST::build(args...))
    {}

    Tensor(Tensor &tensor) = default;
    Tensor &operator=(Tensor &tensor) = default;

    template<typename Idx>
    auto operator[](Idx idx) noexcept
    {
        return at(idx);
    }

    template<int I, bool C, typename Result>
    struct TensorInfo
    {
        static constexpr int index = I;
        static constexpr bool ctg = C;
        size_t offset = 0;
        Result result;
    };

    template<typename... Slices>
    auto at(Slices... args)
    {
        using namespace std;
        constexpr size_t dims = sizeof...(Axes);
        auto slices1 = make_tuple(Slice(args)...);
        constexpr auto axes = make_tuple(Axes()...);
        constexpr size_t news = countIf([] (const auto &s) { return s.kind == ST::New; }, slices1);
        auto slices2 = tupleFill<dims + news>(Slice(), slices1);
        auto [offset, items] = apply([this, &axes] (auto... args) {
            return reduce([this, &axes] (const auto &value, auto slice) {
                constexpr int index = value.index;
                auto &size = get<index>(shape);
                auto &stride = get<index>(strides);
                auto &axis = get<index>(axes);
                if constexpr (slice.kind == ST::Index) {
                    assert(slice.end >= 0 && slice.end < size);
                    return TensorInfo<index - 1, axis.size == 1 && value.ctg, decltype(value.result)>{value.offset + product(slice.end, stride), value.result};
                } else if constexpr(slice.kind == ST::None) {
                    Axis<axis.id, axis.size, value.ctg ? Auto : axis.stride> newAxis;
                    auto result = pushFront(make_tuple(size, stride, newAxis), value.result);
                    return TensorInfo<index - 1, axis.contiguous, decltype(result)>{value.offset, result};
                } else if constexpr(slice.kind == ST::New) {
                    assert(!"!!!");
                    return TensorInfo{value.offset, pushFront(make_tuple(1, 0, Axis<0, 1>()), value.result)};
                } else if constexpr(slice.kind == ST::Slice) {
                    auto size = slice.size();
                    Axis<axis.id, is_same_v<decltype(size), int> ? Dynamic : static_cast<int>(size), value.ctg ? axis.stride : Dynamic> newAxis;
                    auto result = pushFront(make_tuple(size, product(stride, slice.step), newAxis), value.result);
                    return TensorInfo<index - 1, false /*?*/, decltype(result)>{value.offset + product(slice.begin, stride), result};
                }
            }, TensorInfo<dims - 1, true, tuple<>>(), args...);
        }, zip<true>(slices2));
        if constexpr(tuple_size_v<decltype(items)> == 0) {
            return Reference(pointer + offset);
        } else {
            auto [shape, strides, axes] = applyZip(items);
            auto ptr = pointer + offset;
            using NewTensor = Apply<Tensor, decltype(pushFront(ptr, axes))>;
            static_assert(is_convertible_v<decltype(shape), typename NewTensor::ShapeType>, "Wrong shape deduced");
            static_assert(is_convertible_v<decltype(strides), typename NewTensor::StridesType>, "Wrong strides deduced");
            return NewTensor(ptr, shape, strides);
        }
    }

    template<typename Other>
    operator Tensor<Other, Axes...>() const
    {
        Other result(shape);
        return result;
    }

    template<typename OtherPtr, AxisLike... OtherAxes>
    void copyFrom(const Tensor<OtherPtr, OtherAxes...> &other)
    {
        using namespace std;
        static_assert(is_same_v<Item, GetItem<OtherPtr>>, "Item types must be the same");
        static_assert(is_same_v<Ids, typename Tensor<OtherPtr, OtherAxes...>::Ids>, "Tensor dims must be the same");
        static_assert(((Axes::dynamic || OtherAxes::dynamic || Axes::size == OtherAxes::size) && ...), "Tensor shapes must be the same");
        assert(shape == other.shape);

        auto cs = commonStrides<sizeof(Item)>(make_tuple(BoolConst<Axes::contiguous && OtherAxes::contiguous>() ...), shape.tuple(), other.strides, strides);
        size_t size = get<0>(cs);
        apply([this, size, &other] (auto... args) {
            Copy<GetBuffer<OtherPtr>, GetBuffer<Pointer>>::copy(other.rawPointer(), rawPointer(), size, args...);
        }, get<1>(cs));
    }

    template<BufferLike Buffer>
    auto to() const
    {
        AllocatedTensor<Buffer, Item, RemoveStride<Axes>...> result(shape);
        result.copyFrom(*this);
        return result;
    }

    const void *rawPointer() const noexcept { return pointer; }
    void *rawPointer() noexcept { return pointer; }

    friend inline std::ostream& operator<<(std::ostream &stream, Tensor &t) noexcept
    {
        stream << "Tensor(";
        ((stream << Axes()), ...);
        stream << ", shape: " << t.shape.tuple();
        stream << ", strides: " << t.strides;
        return stream << ")";
    }

    const ShapeType shape;
    const StridesType strides;
protected:

    size_t calcOffset(Integer auto... indices)
    {
        using namespace std;
        return apply([] (const auto& ... args) {
            return (apply([] (Integer auto index, Integer auto size, Integer auto stride) {
                if (index < 0 || index > size)
                    throw out_of_range("Index out of range");
                return index * stride;
            }, args) + ...);
        }, zip(make_tuple(indices...), shape.tuple(), strides));
    }

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

using vt::operator<<;

#endif // VT_CORE_H
