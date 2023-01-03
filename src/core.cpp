#include <cstring>
#include <numeric>
#include <execution>

#include "./core.h"

namespace vt {

HeapBuffer::HeapBuffer(size_t size) :
    memory(new uint8_t[size])
{}

void *HeapBuffer::get() noexcept
{
    return memory.get();
}

const void *HeapBuffer::get() const noexcept {
    return memory.get();
}

template<> void copy<PassiveBuffer, HeapBuffer>(const void* src, void *dst, size_t size)
{
    std::memcpy(dst, src, size);
}

template<> void copy<PassiveBuffer, HeapBuffer, std::tuple<int, int, int>>(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides)
{
    using namespace std;
    unique_ptr<size_t[]> indexes(new size_t[rows]);
    iota(indexes.get(), indexes.get() + rows, 0);
    for_each_n(execution::par, indexes.get(), rows, [src, dst, &strides] (size_t n) {
        auto [cols, s, d] = strides;
        memcpy(static_cast<uint8_t *>(dst) + n * d, static_cast<const uint8_t *>(src) + n * s, cols);
    });
}

template<> void copy<PassiveBuffer, PassiveBuffer, std::tuple<int, int, int>>(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides)
{
    copy<PassiveBuffer, HeapBuffer, std::tuple<int, int, int>>(src, dst, rows, strides);
}


template<> void copy<HeapBuffer, PassiveBuffer>(const void* src, void *dst, size_t size)
{
    std::memcpy(dst, src, size);
}

}
