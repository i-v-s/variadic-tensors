#ifndef CUDA_H
#define CUDA_H
#include "./core.h"

namespace vt {

class PinnedBuffer: public Buffer<PinnedBuffer>
{
public:
    using Parent = Buffer<PinnedBuffer>;
    using Parent::Parent;
    constexpr static Device device = Device::Host;
protected:
    static void *malloc(size_t size);
    static void dealloc(void *ptr);
    friend class Buffer<PinnedBuffer>;
};

class CudaBuffer: public Buffer<CudaBuffer>
{
public:
    using Parent = Buffer<CudaBuffer>;
    using Parent::Parent;
    constexpr static Device device = Device::Cuda;
protected:
    static void *malloc(size_t size);
    static void dealloc(void *ptr);
    friend class Buffer<CudaBuffer>;
};

template<typename Item, bool offset>
using CudaPointer = SharedPointer<CudaBuffer, Item, offset>;

template<typename Item, bool offset>
using PinnedPointer = SharedPointer<PinnedBuffer, Item, offset>;

template<typename Item, typename... Args>
using CudaTensor = AllocatedTensor<CudaBuffer, Item, Args...>;

template<typename Item, typename... Args>
using PinnedTensor = AllocatedTensor<PinnedBuffer, Item, Args...>;


struct HostCudaCopy{
    static void copy(const void* src, void *dst, size_t size);
    static void copy(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides);
};

struct CudaHostCopy{
    static void copy(const void* src, void *dst, size_t size);
    static void copy(const void* src, void *dst, size_t rows, const std::tuple<int, int, int> &strides);
};

template<HostBufferLike Src, CudaBufferLike Dst>
struct Copy<Src, Dst> : HostCudaCopy {};

template<CudaBufferLike Src, HostBufferLike Dst>
struct Copy<Src, Dst> : CudaHostCopy {};

}

#endif // CUDA_H
