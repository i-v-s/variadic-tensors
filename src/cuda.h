#ifndef CUDA_H
#define CUDA_H
#include "./core.h"

namespace vt {

class PinnedBuffer: public Buffer<PinnedBuffer>
{
public:
    using Parent = Buffer<PinnedBuffer>;
    using Parent::Parent;
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

}

#endif // CUDA_H
