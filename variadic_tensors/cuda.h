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

    template<typename Src, typename Dst>
    static void resizeBatch(const std::vector<Src> &src, std::vector<Dst> &dst,
                            NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr)
    {
        using namespace std;
        static_assert(is_same_v<typename Src::Item, typename Dst::Item>);
        size_t size = src.size();
        auto &ctx = context();
        ctx.hStream = stream;
        NppiSize minSrc, maxSrc, minDst, maxDst;
        vector<NppiSize> srcSizes, dstSizes;
        if (checkSizes(src, srcSizes, minSrc, maxSrc) || checkSizes(dst, dstSizes, minDst, maxDst)) {
            vector<NppiImageDescriptor> srcDesc(size), dstDesc(size);
            vector<NppiResizeBatchROI_Advanced> roi(size);
            for (int i = 0; i < size; i++) {
                srcDesc[i] = { const_cast<void *>(src[i].rawPointer()), get<0>(src[i].strides()), srcSizes[i] };
                dstDesc[i] = { dst[i].rawPointer(), get<0>(dst[i].strides()), dstSizes[i] };
                roi[i] = { fullRect(srcSizes[i]), fullRect(dstSizes[i]) };
            }
            nppiResizeBatchAdvanced<typename Src::Item, 3>(maxDst, srcDesc, dstDesc, roi, mode, ctx);
        } else {
            vector<NppiResizeBatchCXR> batch(size);
            for (int i = 0; i < size; i++)
                batch[i] = { src[i].rawPointer(), get<0>(src[i].strides()), dst[i].rawPointer(), get<0>(dst[i].strides()) };
            nppiResizeBatch<typename Src::Item, 3>(minSrc, fullRect(minSrc), minDst, fullRect(minDst), mode, batch, ctx);
        }
    }

private:

    template<Integer H, Integer W>
    static NppiSize nppiSize(const std::tuple<H, W, std::integral_constant<int, 3>> &shape) noexcept
    {
        using namespace std;
        return { get<1>(shape), get<0>(shape) };
    }

    static NppiRect fullRect(const NppiSize &size) noexcept
    {
        return { 0, 0, size.width, size.height };
    }

    template<typename Tensor>
    static bool checkSizes(const std::vector<Tensor> &items, std::vector<NppiSize> &sizes, NppiSize &minSize, NppiSize &maxSize) noexcept
    {
        minSize = maxSize = nppiSize(items[0].shape());
        sizes.push_back(minSize);
        for (int i = 1, e = items.size(); i < e; i++) {
            auto [w, h] = nppiSize(items[i].shape());
            sizes.emplace_back(w, h);
            if (w > maxSize.width) maxSize.width = w;
            if (h > maxSize.height) maxSize.height = h;
            if (w < minSize.width) minSize.width = w;
            if (h < minSize.height) minSize.height = h;
        }
        return (minSize.width != maxSize.width || minSize.height != maxSize.height);
    }

    template<typename P, uint C>
    static void nppiResizeBatch(
            NppiSize, NppiRect, NppiSize, NppiRect, NppiInterpolationMode, const std::vector<NppiResizeBatchCXR> &batchList, NppStreamContext);

    template<typename P, uint C>
    static void nppiResizeBatchAdvanced(
            const NppiSize &maxDst, const std::vector<NppiImageDescriptor> &pBatchSrc, const std::vector<NppiImageDescriptor> &pBatchDst, const std::vector<NppiResizeBatchROI_Advanced> &pBatchROI, NppiInterpolationMode eInterpolation, NppStreamContext nppStreamCtx);
};

template<CudaBufferLike Buffer>
struct Resize<Buffer> : public CudaResize {};

}

#endif // VT_CUDA_H
