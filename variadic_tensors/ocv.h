#ifndef OCV_H
#define OCV_H
#include <opencv2/core.hpp>

#include "./axis.h"
#include "./core.h"

namespace vt {

template<typename Item> struct cvTypeT;
template<> struct cvTypeT<uint8_t> { constexpr static int value = CV_8U; };
template<> struct cvTypeT<int8_t> { constexpr static int value = CV_8S; };
template<> struct cvTypeT<uint16_t> { constexpr static int value = CV_16U; };
template<> struct cvTypeT<int16_t> { constexpr static int value = CV_16S; };
template<> struct cvTypeT<int32_t> { constexpr static int value = CV_32S; };
template<> struct cvTypeT<float> { constexpr static int value = CV_32F; };
template<> struct cvTypeT<double> { constexpr static int value = CV_64F; };
template<> struct cvTypeT<cv::float16_t> { constexpr static int value = CV_16F; };

template<typename Item, int channels>
static constexpr int cvType = CV_MAKETYPE(cvTypeT<std::remove_const_t<Item>>::value, channels);

template<HostBufferLike Buffer> struct Export<Buffer, cv::Mat>
{
    template<typename Item, int channels>
    static cv::Mat create(Item *ptr,
                          const std::tuple<int, int, std::integral_constant<int, channels> > &shape,
                          const std::tuple<int, std::integral_constant<int, channels>, std::integral_constant<int, 1> > &strides)
    {
        static_assert(channels >= 1 && channels <= 4, "Wrong channel number");
        return cv::Mat(get<0>(shape), get<1>(shape), cvType<Item, channels>, ptr, get<0>(strides));
    }
};


template<template<typename... Args> typename Tensor, typename Item, int H, int W, int C, int channels>
struct Import<cv::Mat, Tensor<Item *, vt::Axis<H>, vt::Axis<W>, vt::Axis<C, channels>>>
{
    static_assert(channels >= 1 && channels <= 4, "Wrong channel number");
    using Result = PassiveTensor<uint8_t, vt::Axis<H>, vt::Axis<W>, vt::Axis<C, channels>>;
    static Result create(cv::Mat &image)
    {
        if (image.type() != cvType<Item, channels>)
            throw std::runtime_error("cv::Mat type mismatch");
        return Result(image.data, image.rows, image.cols);
    }
};

template<template<typename... Args> typename Tensor, typename Item, int H, int W, int C, int channels>
struct Import<cv::Mat, Tensor<Item *, vt::Axis<H, Dynamic, Dynamic>, vt::Axis<W>, vt::Axis<C, channels>>>
{
    static_assert(channels >= 1 && channels <= 4, "Wrong channel number");
    using Result = PassiveTensor<uint8_t, vt::Axis<H, Dynamic, Dynamic>, vt::Axis<W>, vt::Axis<C, channels>>;
    static Result create(cv::Mat &image)
    {
        if (image.type() != cvType<Item, channels>)
            throw std::runtime_error("cv::Mat type mismatch");
        return Result(image.data, image.rows, image.cols, static_cast<size_t>(image.step));
    }
};

}
#endif // OCV_H
