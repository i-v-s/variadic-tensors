#include <iostream>

#include "src/cuda.h"

using namespace std;

enum Dims {
    Width, Height, Channels, Batch
};

int main(int argc, char *argv[])
{
    vt::Slice sl{vt::Const<0>()};
    bool e = sl.empty();

    using StridedImage = vt::PassiveTensor<uint8_t, vt::Axis<Height, vt::Dynamic, vt::Dynamic>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using HostImage = vt::PinnedTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
    using CudaImage = vt::CudaTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;

    StridedImage si(nullptr, 240, 320, 355);
    si.at(vt::Slice{2, vt::Const<7>()}, 3);
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

    cout << "ok" << endl;
}
