#ifndef VT_CUDA_H
#define VT_CUDA_H
#include <nppdefs.h>

#include "./core.h"

namespace vt {

namespace npp {

NppiRect fullRect(const NppiSize &size) noexcept;
NppStreamContext &context();

template<Integer H, Integer W>
NppiSize nppiSize(const std::tuple<H, W, std::integral_constant<int, 3>> &shape) noexcept
{
    using namespace std;
    return { get<1>(shape), get<0>(shape) };
}

}

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

    static void apply(const uint8_t* src, uint8_t *dst,
                       const std::tuple<int, int, IntConst<3>> &srcShape, const std::tuple<int, int, IntConst<3>> &dstShape,
                       const std::tuple<int, IntConst<3>, IntConst<1>> &srcStrides, const std::tuple<int, IntConst<3>, IntConst<1>> &dstStrides,
                       NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr);

    template<typename Src, typename Dst>
    static void applyBatch(const std::vector<Src> &src, std::vector<Dst> &dst,
                            NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr)
    {
        using namespace std;
        using namespace npp;
        static_assert(is_same_v<typename Src::Item, typename Dst::Item>);
        size_t size = src.size();
        auto &ctx = context();
        ctx.hStream = stream;
        NppiSize minSrc, maxSrc, minDst, maxDst;
        vector<NppiSize> srcSizes, dstSizes;
        bool se = checkSizes(src, srcSizes, minSrc, maxSrc),
             de = checkSizes(dst, dstSizes, minDst, maxDst);
        if (se && de) {
            vector<NppiResizeBatchCXR> batch(size);
            for (int i = 0; i < size; i++)
                batch[i] = { src[i].data(), get<0>(src[i].strides()), dst[i].data(), get<0>(dst[i].strides()) };
            nppiResizeBatch<typename Src::Item, 3>(minSrc, fullRect(minSrc), minDst, fullRect(minDst), mode, batch, ctx);
        } else {
            vector<NppiImageDescriptor> srcDesc(size), dstDesc(size);
            vector<NppiResizeBatchROI_Advanced> roi(size);
            for (int i = 0; i < size; i++) {
                srcDesc[i] = { const_cast<uint8_t *>(src[i].data()), get<0>(src[i].strides()), srcSizes[i] };
                dstDesc[i] = { dst[i].data(), get<0>(dst[i].strides()), dstSizes[i] };
                roi[i] = { fullRect(srcSizes[i]), fullRect(dstSizes[i]) };
            }
            nppiResizeBatchAdvanced<typename Src::Item, 3>(maxDst, srcDesc, dstDesc, roi, mode, ctx);
        }
    }

private:
    /**
     * @brief checkSizes function that check sizes, finds minimal, maximal and fills sizes
     * @param items source tensors
     * @param sizes vector of sizes to fill
     * @param minSize
     * @param maxSize
     * @return true if item sizes are the same
     */
    template<typename Tensor>
    static bool checkSizes(const std::vector<Tensor> &items, std::vector<NppiSize> &sizes, NppiSize &minSize, NppiSize &maxSize) noexcept
    {
        minSize = maxSize = npp::nppiSize(items[0].shape());
        sizes.push_back(minSize);
        for (int i = 1, e = items.size(); i < e; i++) {
            auto [w, h] = npp::nppiSize(items[i].shape());
            sizes.emplace_back(w, h);
            if (w > maxSize.width) maxSize.width = w;
            if (h > maxSize.height) maxSize.height = h;
            if (w < minSize.width) minSize.width = w;
            if (h < minSize.height) minSize.height = h;
        }
        return minSize.width == maxSize.width && minSize.height == maxSize.height;
    }

    template<typename P, uint C>
    static void nppiResizeBatch(
            NppiSize, NppiRect, NppiSize, NppiRect, NppiInterpolationMode, const std::vector<NppiResizeBatchCXR> &batchList, NppStreamContext);

    template<typename P, uint C>
    static void nppiResizeBatchAdvanced(
            const NppiSize &maxDst, const std::vector<NppiImageDescriptor> &pBatchSrc, const std::vector<NppiImageDescriptor> &pBatchDst, const std::vector<NppiResizeBatchROI_Advanced> &pBatchROI, NppiInterpolationMode eInterpolation, NppStreamContext nppStreamCtx);
};

struct CudaWarpAffine
{
    static void apply(
            const uint8_t *src, uint8_t *dst,
            const std::tuple<int, int, IntConst<3>> &srcShape, const std::tuple<int, int, IntConst<3>> &dstShape,
            const std::tuple<int, IntConst<3>, IntConst<1>> &srcStrides, const std::tuple<int, IntConst<3>, IntConst<1>> &dstStrides,
            const AffineMatrix &coeffs, NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr);

    static void applyBatch(
            const std::tuple<int, int, int, IntConst<3> > &srcShape, const std::tuple<int, int, int, IntConst<3> > &dstShape,
            const WarpAffineTask<uint8_t> *tasks, const std::vector<AffineMatrix> &matrices, NppiInterpolationMode mode = NPPI_INTER_LINEAR, cudaStream_t stream = nullptr);
};

template<CudaBufferLike Buffer> struct Resize<Buffer> : public CudaResize {};
template<CudaBufferLike Buffer> struct WarpAffine<Buffer> : public CudaWarpAffine {};

}

#endif // VT_CUDA_H
