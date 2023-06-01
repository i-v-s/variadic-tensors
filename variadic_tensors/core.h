#ifndef VT_CORE_H
#define VT_CORE_H
#include <concepts>
#include <memory>
#include <span>
#include <atomic>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <sstream>
#include <stdexcept>
#include <assert.h>

#include "./axis.h"
#include "./shape.h"
#include "./utils.h"
#include "./strides.h"
#include "./slice.h"
#include "./buffers.h"
#include "./pointers.h"
#include "./actions.h"

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

template<class T>
concept TensorLike =
        requires(T t) {
            { t.shape().tuple() } -> IntegerTupleLike;
            { t.strides() } -> IntegerTupleLike;
            { t.data() } -> std::convertible_to<const void *>;
        };


template<TensorLike Source, TensorLike Target>
void toVector(std::vector<Target> &result, Source &source, size_t limit = 0) noexcept
{
    using namespace std;
    auto size = get<0>(source.shape().tuple());
    assert(limit <= size);
    for (size_t i = 0, e = limit ? limit : size; i < e; i++)
        result.push_back(source[i]);
}

template<typename Pointer_, AxisLike... Axes>
class ConstTensor
{
public:
    using Pointer = Pointer_;
    using Buffer = GetBuffer<Pointer>;
    using SST = ShapeStridesTuple<Axes...>;
    using Ids = std::integer_sequence<int, Axes::id...>;
    using Item = GetItem<Pointer>;
    using ShapeType = Shape<typename SST::Shape>;
    using StridesType = typename SST::Strides;
    static constexpr bool onHost = HostBufferLike<Buffer>;

    ConstTensor() :
        shape_{},
        strides_{},
        pointer(nullptr)
    {}

    ConstTensor(Pointer pointer, const ShapeType &shape) :
        shape_(shape),
        strides_(SST::buildStrides(shape)),
        pointer(pointer)
    {}

    ConstTensor(Pointer pointer, const ShapeType &shape, const StridesType &strides) :
        shape_(shape),
        strides_(strides),
        pointer(pointer)
    {}

    template<std::integral... Args>
    ConstTensor(Pointer pointer, Args... args) : ConstTensor(pointer, SST::build(args...))
    {}

    ConstTensor(const ConstTensor &) = default;

    ConstTensor(ConstTensor && other) :
        shape_(other.shape_),
        strides_(other.strides_),
        pointer(std::move(other.pointer))
    {
        other.pointer = nullptr;
    }

    ConstTensor &operator=(const ConstTensor &) = default;

    ConstTensor &operator=(ConstTensor && other)
    {
        shape_ = other.shape_;
        strides_ = other.strides_;
        pointer = std::move(other.pointer);
        other.pointer = nullptr;
        return *this;
    }

    void reset() noexcept
    {
        pointer = nullptr;
    }

    bool empty() const noexcept
    {
        return !bool(pointer);
    }

    template<typename Other> static ConstTensor from(Other &item)
    {
        return Import<Other, ConstTensor>::create(item);
    }

    template<typename Idx>
    auto operator[](Idx idx) const
    {
        return at(idx);
    }

    template<typename... Slices>
    auto at(Slices && ... args) const
    {
        using namespace std;
        auto [offset, items] = calcAt(forward<Slices>(args)...);
        if constexpr(tuple_size_v<decltype(items)> == 0) {
            return ConstReference(pointer + offset);
        } else {
            auto [shape, strides, axes] = applyZip(items);
            auto ptr = pointer + offset;
            using NewTensor = Apply<ConstTensor, decltype(pushFront(ptr, axes))>;
            static_assert(is_convertible_v<decltype(shape), typename NewTensor::ShapeType>, "Wrong shape deduced");
            static_assert(is_convertible_v<decltype(strides), typename NewTensor::StridesType>, "Wrong strides deduced");
            return NewTensor(ptr, shape, strides);
        }
    }

    auto span() const
    {
        using namespace std;
        static_assert(onHost, "Tensor must be located on host memory");
        static_assert(sizeof...(Axes) == 1, "span available only for one dimensional Tensor");
        assert(get<0>(strides_) == 1);
        const auto &size = get<0>(shape_);
        auto ptr = static_cast<const Item *>(data());
        if constexpr(IntConstLike<decltype(size)>)
            return std::span<Item, decltype(size)::value>(ptr);
        else
            return std::span<Item>(ptr, size);
    }

    template<BufferLike Buffer>
    auto to() const
    {
        AllocatedTensor<Buffer, Item, RemoveStride<Axes>...> result(shape_);
        result.copyFrom(*this);
        return result;
    }

    template<typename Other>
    Other as() const
    {
        return std::apply([p = data()] (auto... args) {
            return Export<Buffer, Other>::create(p, args...);
        }, std::tuple_cat(shape_.tuple(), strides_));
    }

    auto asVector(size_t limit = 0) const noexcept
    {
        std::vector<decltype((*this)[0])> result;
        toVector(result, *this, limit);
        return result;
    }

    template<template<typename T> typename Action, TensorLike Dst, typename... Args>
    void actionTo(Dst &result, Args && ... args) const
    {
        using namespace std;
        static_assert(std::is_same_v<Item, typename Dst::Item>, "Tensor item types must be the same");
        static_assert(Buffer::device == Dst::Buffer::device, "Tensors must be on the same device");
        Action<GetBuffer<typename Dst::Pointer>>::apply(
                    data(), result.data(),
                    shape_, result.shape(), strides_, result.strides(), std::forward<Args>(args)...);
    }

    template<TensorLike Dst, typename... Args>
    void resizeTo(Dst && dst, Args && ... args) const { actionTo<Resize>(static_cast<Dst&>(dst), std::forward<Args>(args)...); }

    template<TensorLike Dst, typename... Args>
    void warpAffineTo(Dst && result, const AffineMatrix &coeffs, Args && ... args) const { actionTo<WarpAffine>(static_cast<Dst&>(result), coeffs, std::forward<Args>(args)...); }

    template<int... ids, typename... Args>
    auto resize(Args... args) const
    {
        using namespace std;
        static_assert(((findIndex<ids, Axes::id...> >= 0) && ...), "Specified non-existent axis id");
        static_assert(sizeof...(ids) <= sizeof...(args), "Count of ids must be equal to count of sizes");
        constexpr auto indices = make_tuple(findIndex<Axes::id, ids...>...);
        auto argsTuple = forward_as_tuple(args...);
        auto sizesTuple = tupleSlice<0, sizeof...(ids)>(argsTuple);

        auto newShape = tupleMap<true>([&indices, &sizesTuple] <int i> (Integer auto size) {
            constexpr int idx = get<i>(indices);
            if constexpr(idx >= 0)
                return get<idx>(sizesTuple);
            else
                return size;
        }, zip(shape_.tuple()));
        // TODO: Fix axes
        using Dst = AllocatedTensor<GetBuffer<Pointer>, Item, RemoveStride<Axes>...>;
        Dst result(newShape);
        apply([this, &result] (auto... args) { resizeTo(result, args...); }, tupleSlice<sizeof...(ids)>(argsTuple));
        return result;
    }

    const Item *data() const noexcept { return pointer; }

    friend inline std::ostream& operator<<(std::ostream &stream, const ConstTensor &t) noexcept
    {
        stream << "Tensor(";
        ((stream << Axes()), ...);
        stream << ", shape: " << t.shape().tuple();
        stream << ", strides: " << t.strides;
        return stream << ")";
    }

    template<uint id>
    auto shape() const noexcept {
        constexpr int idx = findIndex<id, Axes::id...>;
        static_assert(idx >= 0, "Axis id not found");
        return get<idx>(shape_.tuple());
    }
    const ShapeType &shape() const noexcept { return shape_; };
    const StridesType &strides() const noexcept { return strides_; };

    constexpr static bool contiguous = (Axes::contiguous && ...);

protected:

    ConstTensor(Pointer pointer, typename SST::Pair && sst) :
        shape_(sst.first),
        strides_(sst.second),
        pointer(pointer)
    {}

    template<int I, bool C, typename Result>
    struct TensorInfo
    {
        static constexpr int index = I;
        static constexpr bool ctg = C;
        size_t offset = 0;
        Result result;
    };

    template<typename... Slices>
    auto calcAt(Slices && ... args) const
    {
        using namespace std;
        constexpr size_t dims = sizeof...(Axes);
        auto slices1 = make_tuple(Slice(args)...);
        constexpr auto axes = make_tuple(Axes()...);
        constexpr size_t news = countIf([] (const auto &s) { return s.kind == ST::New; }, slices1);
        auto slices2 = tupleFill<dims + news>(Slice(), slices1);
        return apply([this, &axes] (auto... args) {
            return reduce([this, &axes] (const auto &value, auto slice) {
                constexpr int index = value.index;
                auto &size = get<index>(shape_);
                auto &stride = get<index>(strides_);
                auto &axis = get<index>(axes);
                if constexpr (slice.kind == ST::Index) {
                    if (slice.end < 0 || slice.end >= size)
                        throw out_of_range("at() argument " + to_string(slice.end) + " not in range 0.." + to_string(size));
                    return TensorInfo<index - 1, axis.size == 1 && value.ctg, decltype(value.result)>{value.offset + product(slice.end, stride), value.result};
                } else if constexpr(slice.kind == ST::None) {
                    Axis<axis.id, axis.size, value.ctg ? Auto : axis.stride> newAxis;
                    auto result = pushFront(make_tuple(size, stride, newAxis), value.result);
                    return TensorInfo<index - 1, axis.contiguous, decltype(result)>{value.offset, result};
                } else if constexpr(slice.kind == ST::New) {
                    assert(!"!!!");
                    return TensorInfo{value.offset, pushFront(make_tuple(1, 0, Axis<0, 1>()), value.result)};
                } else if constexpr(slice.kind == ST::Slice) {
                    if (slice.begin < 0 || slice.begin >= size)
                        throw out_of_range("at() slice begin " + to_string(slice.begin) + " not in range 0.." + to_string(size));
                    auto size = slice.size();
                    Axis<axis.id, is_same_v<decltype(size), int> ? Dynamic : static_cast<int>(size), value.ctg ? axis.stride : Dynamic> newAxis;
                    auto result = pushFront(make_tuple(size, product(stride, slice.step), newAxis), value.result);
                    return TensorInfo<index - 1, false /*?*/, decltype(result)>{value.offset + product(slice.begin, stride), result};
                }
            }, TensorInfo<dims - 1, true, tuple<>>(), args...);
        }, zip<true>(slices2));
    }

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

    ShapeType shape_;
    StridesType strides_;
    mutable Pointer pointer;
};

template<typename Pointer, AxisLike... Axes>
class Tensor: public ConstTensor<Pointer, Axes...>
{
public:
    using Const = ConstTensor<Pointer, Axes...>;
    using Buffer = typename Const::Buffer;
    using ShapeType = typename Const::ShapeType;
    using StridesType = typename Const::StridesType;
    using Item = typename Const::Item;
    using Ids = typename Const::Ids;

    Tensor()
    {}

    Tensor(Pointer pointer, const ShapeType &shape) :
        Const(pointer, shape)
    {}

    Tensor(Pointer pointer, const  ShapeType &shape, const StridesType &strides) :
        Const(pointer, shape, strides)
    {}

    template<std::integral... Args>
    Tensor(Pointer pointer, Args... args) :
        Const(pointer, Const::SST::build(std::forward<Args>(args)...))
    {}

    using Const::operator=;
    Tensor &operator=(const Const &tensor) = delete;
    Tensor &operator=(Const && other) = delete;

    template<typename Other> static Tensor from(Other &item)
    {
        return Import<Other, Tensor>::create(item);
    }

    template<typename Other>
    Other as()
    {
        return std::apply([p = data()] (auto... args) {
            return Export<Buffer, Other>::create(p, args...);
        }, std::tuple_cat(Const::shape_.tuple(), Const::strides_));
    }

    using Const::asVector;
    auto asVector(size_t limit = 0) noexcept
    {
        std::vector<decltype((*this)[0])> result;
        toVector(result, *this, limit);
        return result;
    }

    using Const::operator[];

    template<typename Idx> auto operator[](Idx idx)
    {
        return at(idx);
    }

    using Const::at;

    template<typename... Slices>
    auto at(Slices && ... args)
    {
        using namespace std;
        auto [offset, items] = Const::calcAt(forward<Slices>(args)...);
        if constexpr(tuple_size_v<decltype(items)> == 0) {
            return Reference(Const::pointer + offset);
        } else {
            auto [shape, strides, axes] = applyZip(items);
            auto ptr = Const::pointer + offset;
            using NewTensor = Apply<Tensor, decltype(pushFront(ptr, axes))>;
            static_assert(is_convertible_v<decltype(shape), typename NewTensor::ShapeType>, "Wrong shape deduced");
            static_assert(is_convertible_v<decltype(strides), typename NewTensor::StridesType>, "Wrong strides deduced");
            return NewTensor(ptr, shape, strides);
        }
    }

    Tensor &operator=(std::initializer_list<Item> items)
    {
        static_assert(Const::onHost, "Tensor must be located on host memory");
        static_assert(sizeof...(Axes) == 1, "= available only for one dimensional Tensor");
        assert(items.size() == get<0>(Const::shape_));
        auto ptr = data();
        auto stride = get<0>(Const::strides_);
        for (const auto &item: items) {
            *ptr = item;
            ptr += stride;
        }
        return *this;
    }

    using Const::span;

    auto span()
    {
        using namespace std;
        static_assert(Const::onHost, "Tensor must be located on host memory");
        static_assert(sizeof...(Axes) == 1, "span available only for one dimensional Tensor");
        assert(get<0>(Const::strides) == 1);
        const auto &size = get<0>(Const::shape);
        auto ptr = data();
        if constexpr(IntConstLike<decltype(size)>)
            return std::span<Item, remove_reference_t<decltype(size)>::value>(ptr, size);
        else
            return std::span<Item>(ptr, size);
    }

    template<typename OtherPtr, AxisLike... OtherAxes>
    void copyFrom(const ConstTensor<OtherPtr, OtherAxes...> &other)
    {
        using namespace std;
        static_assert(is_same_v<Item, GetItem<OtherPtr>>, "Item types must be the same");
        static_assert(is_same_v<Ids, typename ConstTensor<OtherPtr, OtherAxes...>::Ids>, "Tensor dims must be the same");
        static_assert(((Axes::dynamic || OtherAxes::dynamic || Axes::size == OtherAxes::size) && ...), "Tensor shapes must be the same");
        assert(Const::shape() == other.shape());

        auto cs = commonStrides<sizeof(Item)>(make_tuple(BoolConst<Axes::contiguous && OtherAxes::contiguous>() ...), Const::shape().tuple(), other.strides(), Const::strides_);
        size_t size = get<0>(cs);
        apply([this, size, &other] (auto... args) {
            Copy<GetBuffer<OtherPtr>, GetBuffer<Pointer>>::copy(other.data(), data(), size, args...);
        }, get<1>(cs));
    }

    using Const::data;
    Item *data() noexcept { return Const::pointer; }
};

template<template<typename T> typename Action, TensorLike Source, TensorLike Destination, typename... Args>
void actionBatch(const std::vector<Source> &sources, std::vector<Destination> & targets, Args... args)
{
    using namespace std;
    using Buffer = GetBuffer<typename Source::Pointer>;
    static_assert(Buffer::device == Destination::Buffer::device, "Tensors must be on the same device");
    if (sources.size() != targets.size())
        throw runtime_error("vt::actionBatch(): Batches must be the same");
    if (sources.empty())
        return;
    if (sources.size() == 1)
        sources[0].template actionTo<Action>(targets[0], forward<Args>(args)...);
    else
        Action<Buffer>::applyBatch(sources, static_cast<std::vector<Destination>&>(targets), forward<Args>(args)...);
}

template<template<typename T> typename Action, TensorLike Source, TensorLike Destination, typename... Args>
void actionBatch(const std::vector<Source> &sources, Destination &target, Args... args)
{
    using namespace std;
    int dstBatch = get<0>(target.shape());
    if (dstBatch < sources.size())
        throw runtime_error("vt::actionBatch(): Destination batch not enough");
    auto tv = target.asVector(sources.size());
    actionBatch<Action>(sources, tv);
}

template<typename Source, typename Destination, typename... Args>
void resizeBatch(const Source &source, Destination &dst, Args && ... args)
{
    actionBatch<Resize>(source, dst, std::forward<Args>(args)...);
}

template<typename Source, typename Destination, typename... Args>
void warpAffineBatch(const std::vector<Source> &sources, Destination &target, const std::vector<AffineMatrix> &matrices, Args... args)
{
    using namespace std;
    using Buffer = GetBuffer<typename Source::Pointer>;
    using Item = typename Source::Item;
    static_assert(Buffer::device == Destination::Buffer::device, "Tensors must be on the same device");
    static_assert(std::is_same_v<Item, typename Destination::Item>, "Tensor item types must be the same");
    int dstBatch = get<0>(target.shape());
    int batchSize = sources.size();
    if (batchSize != matrices.size() || dstBatch < sources.size())
        throw runtime_error("vt::warpAffineBatch(): Wrong vector sizes");
    if (!batchSize) return;
    if (batchSize == 1)
        sources[0].warpAffineTo(target[0], matrices[0], std::forward<Args>(args)...);
    else {
        auto targetShape = pushFront(batchSize, tupleTail(target.shape().tuple()));
        auto sourceShape = pushFront(batchSize, sources[0].shape().tuple());
        vector<WarpAffineTask<Item>> tasks(batchSize);
        auto targetStride = get<1>(target.strides());
        for (int i = 0, e = batchSize; i < e; i++) {
            assert(tupleTail(sourceShape) == sources[i].shape().tuple());
            tasks[i] = {
                sources[i].data(), get<0>(sources[i].strides()),
                target[i].data(), targetStride
            };
        }
        WarpAffine<Buffer>::applyBatch(sourceShape, targetShape, tasks.data(), matrices, std::forward<Args>(args)...);
    }
}

template<typename Item, typename... Args>
using PassiveTensor = Tensor<Item *, Args...>;

template<BufferLike Buffer, typename Item_, typename... TensorArgs>
class AllocatedTensor: public Tensor<SharedPointer<Buffer, Item_, false>, TensorArgs...>
{
public:
    using Pointer = SharedPointer<Buffer, Item_, false>;
    using Parent = Tensor<Pointer, TensorArgs...>;
    using ShapeType = typename Parent::ShapeType;
    using StridesType = typename Parent::StridesType;
    using Item = typename Parent::Item;

    AllocatedTensor() {}

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
