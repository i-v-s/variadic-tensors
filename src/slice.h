#ifndef SLICE_H
#define SLICE_H
#include <initializer_list>
#include <ostream>

#include "./utils.h"

namespace vt {

enum class ST
{
    None,
    New,
    Index,
    Slice
};

template<ST st> using STConst = std::integral_constant<ST, st>;

template <ST st, Integer Begin, Integer End, Integer Step = std::integral_constant<int, 1>>

struct Slice
{
public:
    Slice(Begin begin, End end) :
        begin(begin), end(end)
    {}

    Slice(Begin begin, End end, Step step) :
        begin(begin), end(end), step(step)
    {}

    Slice(End end) :
        end(end)
    {}

    Slice() {}

    Slice(const Slice &) = default;

    End operator()() {
        static_assert(kind == ST::Index);
        return end;
    }

    constexpr bool empty() const noexcept { return begin == end; }
    constexpr auto size() const noexcept {
        static_assert(kind == ST::Slice);
        return add<1>(div(sub(add<-1>(end), begin), step));
    }
    constexpr int newId() const noexcept {
        static_assert(kind == ST::New);
        return end;
    }

    Begin begin;
    End end;
    Step step = [] { if constexpr(IntConstLike<Step>) return Step(); else return 1; }();
    constexpr static ST kind = st;// = [] { if constexpr(IntConstLike<SliceKind>) return SliceKind(); else return ST::Slice; }();

    friend std::ostream& operator<<(std::ostream &stream, Slice slice) noexcept
    {
        switch (slice.kind) {
        case ST::New: return stream << "New";
        case ST::None: return stream << ":";
        case ST::Index: return stream << slice.end;
        case ST::Slice: {
            if (slice.begin) stream << slice.begin;
            stream << ":";
            stream << slice.end;
            if (slice.step != 1)
                stream << ":" << slice.step;
        }}
        return stream;
    }
};

template<Integer Begin, Integer End>
Slice(Begin, End) -> Slice<ST::Slice, Begin, End, std::integral_constant<int, 1>>;

template<Integer End>
Slice(End) -> Slice<ST::Index, std::integral_constant<int, 0>, End, std::integral_constant<int, 1>>;

Slice() -> Slice<ST::None, std::integral_constant<int, 0>, std::integral_constant<int, 0>, std::integral_constant<int, 1>>;

template<int id>
using NewDim = Slice<ST::New, std::integral_constant<int, 0>, std::integral_constant<int, id>, std::integral_constant<int, 1>>;

//template<std::convertible_to<ST> St>
//Slice(St) -> Slice<St, std::integral_constant<int, 0>, std::integral_constant<int, 0>, std::integral_constant<int, 1>>;

template<class T>
Slice(std::initializer_list<T>) -> Slice<ST::Slice, T, T, T>;

/*struct SliceStruct: public vt::Slice<vt::ST, int, int, int>
{
    using Parent = vt::Slice<vt::ST, int, int, int>;
    SliceStruct(int begin, int end, int step = 1) :
        Parent(begin, end, step)
    {
        kind = vt::ST::Slice;
    }

    SliceStruct(int end) :
        Parent(0, end, 1)
    {
        kind = vt::ST::Index;
    }
    SliceStruct(vt::ST kind)
    {
        this->kind = kind;
    }
};*/

}
#endif // SLICE_H
