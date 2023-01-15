#ifndef VT_ACTIONS_H
#define VT_ACTIONS_H
#include "./buffers.h"

namespace vt {

template<BufferLike SrcBuffer, BufferLike DstBuffer> struct Copy;
template<BufferLike Buffer> struct Resize;
template<BufferLike Buffer, typename Other> struct Export;

struct HostCopy
{
    static void copy(const void* src, void *dst, size_t size);
    static void copy(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides);
};

template<HostBufferLike Src, HostBufferLike Dst>
struct Copy<Src, Dst> : HostCopy {};

}
#endif // VT_ACTIONS_H
