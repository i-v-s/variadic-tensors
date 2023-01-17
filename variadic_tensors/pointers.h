#ifndef VT_POINTERS_H
#define VT_POINTERS_H
#include "./buffers.h"

namespace vt {

template<BufferLike Buffer, typename Item, bool offset = true> class SharedPointer;

template<BufferLike Buffer_, typename Item_>
class SharedPointer<Buffer_, Item_, false>
{
public:
    using Item = Item_;
    using Buffer = Buffer_;

    explicit SharedPointer(size_t size) :
        value(new Buffer(size))
    {}

    SharedPointer(SharedPointer && other) :
        value(std::move(other.value))
    {}

    SharedPointer(std::shared_ptr<Buffer> value) :
        value(value)
    {}

    SharedPointer(const SharedPointer & other) :
        value(other.value)
    {}

    SharedPointer &operator=(std::nullptr_t) noexcept
    {
        value.reset();
        return *this;
    }

    operator Item*() noexcept
    {
        return static_cast<Item *>(value->get());
    }

    operator Item*() const noexcept
    {
        return static_cast<Item *>(value->get());
    }

    operator bool() const noexcept
    {
        return value;
    }

    SharedPointer<Buffer_, Item_, true> operator+(size_t offset)
    {
        return {value, static_cast<Item *>(value->get()) + offset};
    }

protected:
    std::shared_ptr<Buffer> value;
};

template<BufferLike Buffer_, typename Item_>
class SharedPointer<Buffer_, Item_, true>: public SharedPointer<Buffer_, Item_, false>
{
public:
    using Parent = SharedPointer<Buffer_, Item_, false>;
    using Item = typename Parent::Item;

    using Parent::Parent;

    SharedPointer(std::shared_ptr<Buffer_> value, Item_ *ptr) :
        Parent(value), ptr(ptr)
    {}

    operator Item*() noexcept { return ptr; }

    operator Item*() const noexcept { return ptr; }

    SharedPointer<Buffer_, Item_, true> operator+(size_t offset)
    {
        return {Parent::value, ptr + offset};
    }

protected:
    Item_ *ptr;
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




}
#endif // VT_POINTERS_H
