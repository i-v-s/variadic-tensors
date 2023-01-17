#ifndef SHAPE_H
#define SHAPE_H
#include <concepts>
#include <tuple>
#include <utility>

namespace vt {

template<typename Tuple_>
struct Shape: public Tuple_
{
    using Tuple = Tuple_;

    Shape() :
        Tuple{}
    {}

    template<std::integral... Args>
    Shape(const Tuple &v) :
        Tuple(v)
    {}

    Shape(Tuple && v) :
        Tuple(std::forward<Tuple>(v))
    {}

    Tuple &tuple() noexcept
    {
        return *static_cast<Tuple *>(this);
    }

    const Tuple &tuple() const noexcept
    {
        return *static_cast<const Tuple*>(this);
    }

    static constexpr std::size_t size() noexcept
    {
        return std::tuple_size_v<Tuple>;
    }

    template<typename F> inline auto apply(F f) const noexcept
    {
        return std::apply(f, static_cast<Tuple>(*this));
    }

    std::array<int, size()> array() const noexcept
    {
        return apply([] (auto... args) { return std::array<int, size()>{static_cast<int>(args)...}; });
    }

    auto tail() const noexcept
    {
        return shapeTail(*this);
    }

    std::size_t total() const noexcept
    {
        return apply([] (auto... args) { return (args * ...); });
    }
};

}
#endif // SHAPE_H
