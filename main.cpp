#include <iostream>

#include "src/cuda.h"

using namespace std;

enum Dims {
    Width, Height, Channels, Batch
};

template<typename Axes, vt::Integer PrevSize>
struct StridesTupleT;

template<typename... Args> struct Tuple;

template<typename... Axes, typename Axis, vt::Integer PrevSize>
//template<AxisLike... Axes, AxisLike Axis, Integer PrevSize>
struct StridesTupleT<Tuple<Axis, Axes...>, PrevSize>
{
    using Axis1 = vt::Axis<5>;
    /*using Stride = std::conditional_t<Axis::contiguous, PrevSize, typename Axis::Stride>;
    using Size = tx::Product<typename Axis::Size, Stride>;*/
    //using Result = tx::TupleAdd<typename StridesTupleT<std::tuple<Axes...>, PrevSize>::Result, Axis>;
};

template<typename PrevSize>
//template<AxisLike... Axes, AxisLike Axis, Integer PrevSize>
struct StridesTupleT<std::tuple<>, PrevSize>
{
    using Result = std::tuple<>;
};

int main(int argc, char *argv[])
{
    using StridedImage = vt::PassiveTensor<uint8_t, vt::Axis<Height, vt::Dynamic, vt::Dynamic>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using HostImage = vt::PinnedTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using CudaImage = vt::CudaTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;

    StridedImage si(nullptr, 240, 320, 355);
    auto hi = si.to<vt::HeapBuffer>();
    auto ci = si.to<vt::CudaBuffer>();

    CudaImage ci1(si.shape);

    cout << si.shape << endl;
    cout << si.strides << endl;
    cout << ci.shape << endl;
    cout << ci.strides << endl;
    cout << ci1.shape << endl;
    cout << ci1.strides << endl;
    cout << hi.shape << endl;
    cout << hi.strides << endl;

    bool b = std::is_pointer_v<uint8_t *>;
    using pp = std::remove_pointer_t<uint8_t *>;
    pp p = 6;
    int n = sizeof(pp);

    //bool sc = ci.strides.staticContiguous();
    // cout << integral_constant<int, 6>() << endl;
    // cout << tx::Empty() << endl;
    //cout << make_tuple(1, 2, 3) << endl;

    //int sz = tuple_size<CudaImage::CatType>::value;

    //ImageShape tp(480, 640);
    //tx::CudaTensor<uint8_t, tx::Axis<Height>, tx::Axis<Width>, tx::Axis<Channels, 3>> ts(480, 640);

    cout << "ok" << endl;
}
