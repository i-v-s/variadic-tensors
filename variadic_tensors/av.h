#ifndef VT_AV_H
#define VT_AV_H
extern "C" {
#include <libavutil/frame.h>
}

#include "./core.h"

namespace vt {

template<template<typename... Args> typename Tensor, typename Item, int H, int W, int C>
struct Import<AVFrame, Tensor<Item *, vt::Axis<H, Dynamic, Dynamic>, vt::Axis<W>, vt::Axis<C, 3>>>
{
    using Result = Tensor<Item *, vt::Axis<H, Dynamic, Dynamic>, vt::Axis<W>, vt::Axis<C, 3>>;
    static const Result create(const AVFrame &frame)
    {
        if (frame.format != AV_PIX_FMT_BGR24)
            throw std::runtime_error("Wrong pixel format");
        return {frame.data[0], frame.height, frame.width, frame.linesize[0]};
    }
};

}
#endif // VT_AV_H
