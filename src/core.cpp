#include <cstring>
#include <numeric>
#include <execution>

#include "./core.h"

namespace vt {

void HostCopy::copy(const void *src, void *dst, size_t size)
{
    std::memcpy(dst, src, size);
}

void HostCopy::copy(const void *src, void *dst, size_t rows, const std::tuple<int, int, int> &strides)
{
    using namespace std;
    unique_ptr<size_t[]> indexes(new size_t[rows]);
    iota(indexes.get(), indexes.get() + rows, 0);
    for_each_n(execution::par, indexes.get(), rows, [src, dst, &strides] (size_t n) {
        auto [cols, s, d] = strides;
        memcpy(static_cast<uint8_t *>(dst) + n * d, static_cast<const uint8_t *>(src) + n * s, cols);
    });
}

}
