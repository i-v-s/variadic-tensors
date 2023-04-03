#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "variadic_tensors/cuda.h"
#include "variadic_tensors/ocv.h"

using namespace std;

enum Dims {
    Width, Height, Channels, Batch
};

using StridedImage = vt::PassiveTensor<uint8_t, vt::Axis<Height, vt::Dynamic, vt::Dynamic>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
using LocalImage = vt::PassiveTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
using HostImage = vt::PinnedTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;
using CudaImage = vt::CudaTensor<uint8_t, vt::Axis<Height>, vt::Axis<Width>, vt::Axis<Channels, 3>>;


int main(int argc, char *argv[])
{
    cv::Mat image = cv::imread("/media/cv/D/data/cars/good/ambulance/0048.jpg");
    cv::imshow("Image", image);

    LocalImage lic;
    bool e1 = lic.empty();
    lic = LocalImage::from(image);
    bool e2 = lic.empty();

    static_assert(vt::BufferLike<vt::PassiveBuffer>);
    auto licc = lic.to<vt::CudaBuffer>();
    auto liccc = licc.at(vt::Slice(100, 200), vt::Slice(200, 400));
    auto liccc1 = liccc;
    cv::imshow("Crop1", liccc1.resize<Width, Height>(500, 300).to<vt::PinnedBuffer>().as<cv::Mat>());

    LocalImage::Const lic2 = move(lic);
    bool e3 = lic.empty();
    cout << "e1: " << e1 << "; e2: " << e2 << "; e3: " << e3 << endl;

    auto li = LocalImage::from(image);
    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j)
            li.at(i + 100, j + 200) = {128, 255, 0};

    cv::imshow("Modified", image);

    cv::Mat crop = image(cv::Rect(80, 100, 200, 150));
    cv::imshow("Crop", crop);
    cv::MatStep step = crop.step;
    StridedImage cropT(crop.ptr(0, 0), crop.rows, crop.cols, step.buf[0]);
    auto cudaCrop = cropT.to<vt::CudaBuffer>();
    auto cudaR = cudaCrop.resize<Width, Height>(320, 240);
    auto cropH = cudaCrop.to<vt::HeapBuffer>();
    auto cropRH = cudaR.to<vt::HeapBuffer>();

    cv::Mat cropHM(get<0>(cropH.shape()), get<1>(cropH.shape()), CV_8UC3, cropH.rawPointer());
    cv::Mat cropRHM(get<0>(cropRH.shape()), get<1>(cropRH.shape()), CV_8UC3, cropRH.rawPointer());
    cv::imshow("Crop HM", cropHM);
    cv::imshow("Crop RHM", cropRHM);

    cv::waitKey(-1);

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

    cout << si.shape() << endl;
    cout << si.strides() << endl;
    cout << si1.shape() << endl;
    cout << si1.strides() << endl;
    cout << ci.shape() << endl;
    cout << ci.strides() << endl;
    cout << ci1.shape() << endl;
    cout << ci1.strides() << endl;
    //cout << hi.shape << endl;
    //cout << hi.strides << endl;

    cout << "ok" << endl;
}
