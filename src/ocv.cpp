#include "./ocv.h"

using namespace std;

cv::Mat vt::ExportCV::create(uint8_t *ptr, const std::tuple<int, int, std::integral_constant<int, 3> > &shape,
                             const std::tuple<int, std::integral_constant<int, 3>, std::integral_constant<int, 1> > &strides)
{
    return cv::Mat(get<0>(shape), get<1>(shape), CV_8UC3, ptr, get<0>(strides));
}
