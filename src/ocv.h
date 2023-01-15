#ifndef OCV_H
#define OCV_H
#include <opencv2/core.hpp>

#include "./buffers.h"
#include "./actions.h"

namespace vt {

struct ExportCV
{
    static cv::Mat create(uint8_t *ptr,
                          const std::tuple<int, int, std::integral_constant<int, 3>> &shape,
                          const std::tuple<int, std::integral_constant<int, 3>, std::integral_constant<int, 1>> &strides);
};

template<HostBufferLike Buffer>
struct Export<Buffer, cv::Mat> : public ExportCV {};

}
#endif // OCV_H
