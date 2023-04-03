#ifndef VT_CUDA_H
#define VT_CUDA_H
#include <nppdefs.h>

#include "./core.h"

namespace vt {

class PinnedBuffer: private Static
{
public:
    constexpr static Device device = Device::Host;
    static void *malloc(size_t size);
    static void dealloc(void *ptr);
};

class CudaBuffer: private Static
{
public:
    constexpr static Device device = Device::Cuda;
    static void *malloc(size_t size);
    static void dealloc(void *ptr);
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
struct Copy<Src, Dst> : public HostCudaCopy {};

template<CudaBufferLike Src, HostBufferLike Dst>
struct Copy<Src, Dst> : public CudaHostCopy {};

struct CudaResize{
    static NppStreamContext &context();
    static void resize(const uint8_t* src, uint8_t *dst,
                       const std::tuple<int, int, IntConst<3>> &srcShape, const std::tuple<int, int, IntConst<3>> &dstShape,
                       const std::tuple<int, IntConst<3>, IntConst<1>> &srcStrides, const std::tuple<int, IntConst<3>, IntConst<1>> &dstStrides,
                       NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr);
};

template<CudaBufferLike Buffer>
struct Resize<Buffer> : public CudaResize {};

}

#endif // VT_CUDA_H
