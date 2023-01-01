#include <iostream>

#include "src/cuda.h"

using namespace std;

enum Dims {
    Width, Height, Channels, Batch
};

int main(int argc, char *argv[])
{
    using StridedImage = vt::PassiveTensor<uint8_t, vt::Axis<Height, vt::Dynamic, vt::Dynamic>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using LocalImage = vt::PassiveTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using HostImage = vt::PinnedTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using CudaImage = vt::CudaTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;

    array<uint8_t, 5 * 10 * 3> data = {0};

    LocalImage si(data.data(), 5, 10);

    for (int i = 0; i < 3; i++)
        si.at(0, 0, i) = i + 1;

    auto si1 = si;
    si.at(vt::Slice{2, vt::IntConst<7>()}, 3);

    auto ci = si.to<vt::CudaBuffer>();

    auto ai = ci.to<vt::HeapBuffer>();

    array<uint8_t, 5 * 10 * 3> data1;
    LocalImage si11(data1.data(), 5, 10);
    si11.copyFrom(ai);

    auto ci1 = ci;

    cout << si.shape << endl;
    cout << si.strides << endl;
    cout << si1.shape << endl;
    cout << si1.strides << endl;
    cout << ci.shape << endl;
    cout << ci.strides << endl;
    cout << ci1.shape << endl;
    cout << ci1.strides << endl;
    //cout << hi.shape << endl;
    //cout << hi.strides << endl;

    cout << "ok" << endl;
}
