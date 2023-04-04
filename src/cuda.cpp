#include <string>
#include <filesystem>
#include <source_location>
#include <cuda_runtime.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi_geometry_transforms.h>

#include "./cuda.h"

using namespace std;

namespace vt {

inline std::ostream& operator<<(std::ostream &stream, const std::source_location &sl) noexcept
{
    std::filesystem::path const fn = sl.file_name();
    return stream << fn.filename().string() << ":" << std::dec << sl.line()
                  << " [" << sl.function_name() << "]: ";
}

void cudaCheck(cudaError_t status, const std::string &name, std::source_location const& sl = std::source_location::current())
{
    if(status != cudaSuccess) {
        std::ostringstream ss;
        ss << sl << name << " error: " << status << " " << cudaGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}

void cudaCheck(NppStatus status, const std::string &name, std::source_location const& sl = std::source_location::current())
{
    if(status != NPP_NO_ERROR) {
        std::ostringstream ss;
        ss << sl << name << " error: " << status;
        throw std::runtime_error(ss.str());
    }
}

void *CudaBuffer::malloc(size_t size)
{
    void *ptr;
    cudaCheck(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void CudaBuffer::dealloc(void *ptr)
{
    cudaCheck(cudaFree(ptr), "cudaFree");
}

std::ostream &operator<<(std::ostream &stream, const Empty &t) noexcept
{
    return stream << "Empty";
}

void *PinnedBuffer::malloc(size_t size)
{
    void *ptr;
    cudaCheck(cudaHostAlloc(&ptr, size, cudaHostAllocDefault), "cudaHostAlloc");
    return ptr;
}

void PinnedBuffer::dealloc(void *ptr)
{
    cudaCheck(cudaFreeHost(ptr), "cudaFreeHost");
}

void HostCudaCopy::copy(const void *src, void *dst, size_t size)
{
    cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");
}

void HostCudaCopy::copy(const void *src, void *dst, size_t rows, const std::tuple<int, int, int> &strides)
{
    auto [width, s, d] = strides;
    cudaCheck(cudaMemcpy2D(dst, d, src, s, width, rows, cudaMemcpyHostToDevice), "cudaMemcpy2D HostToDevice");
}

void CudaHostCopy::copy(const void *src, void *dst, size_t size)
{
    cudaCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");
}

void CudaHostCopy::copy(const void *src, void *dst, size_t rows, const std::tuple<int, int, int> &strides)
{
    auto [width, s, d] = strides;
    cudaCheck(cudaMemcpy2D(dst, d, src, s, width, rows, cudaMemcpyDeviceToHost), "cudaMemcpy2D DeviceToHost");
}

NppStreamContext &CudaResize::context()
{
    thread_local NppStreamContext ctx = [] {
        NppStreamContext ctx;
        cudaCheck(nppGetStreamContext(&ctx), "nppGetStreamContext");
        return ctx;
    }();
    return ctx;
}

void CudaResize::resize(const uint8_t *src, uint8_t *dst,
                        const std::tuple<int, int, IntConst<3> > &srcShape, const std::tuple<int, int, IntConst<3> > &dstShape,
                        const std::tuple<int, IntConst<3>, IntConst<1> > &srcStrides, const std::tuple<int, IntConst<3>, IntConst<1> > &dstStrides,
                        NppiInterpolationMode mode, cudaStream_t stream)
{
    auto [sh, sw, _1] = srcShape;
    auto [dh, dw, _2] = dstShape;
    NppiRect srcRoi{0, 0, sw, sh}, dstRoi{0, 0, dw, dh};
    auto &ctx = context();
    ctx.hStream = stream;

    cudaCheck(nppiResize_8u_C3R_Ctx(src, get<0>(srcStrides), {sw, sh}, srcRoi,
                                    dst, get<0>(dstStrides), {dw, dh}, dstRoi, mode, ctx),
              "nppiResize_8u_C3R_Ctx");
}

template<typename T>
auto toCuda(const vector<T> &data, cudaStream_t stream = nullptr)
{
    T *ptr;
    size_t size = data.size() * sizeof(T);
    cudaCheck(cudaMalloc(&ptr, size), "cudaMalloc");
    unique_ptr<T, decltype(&cudaFree)> result { ptr, &cudaFree };
    cudaCheck(cudaMemcpyAsync(ptr, data.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    return result;
}

template<>
void vt::CudaResize::nppiResizeBatch<uint8_t, 3>(
        NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, NppiInterpolationMode eInterpolation, const vector<NppiResizeBatchCXR> &batchList, NppStreamContext nppStreamCtx)
{
    auto cudaBatch = toCuda(batchList, nppStreamCtx.hStream);
    cudaCheck(
        nppiResizeBatch_8u_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oSmallestDstSize, oDstRectROI, eInterpolation, cudaBatch.get(), batchList.size(), nppStreamCtx),
        "nppiResizeBatch_8u_C3R_Ctx");
}

template<>
void vt::CudaResize::nppiResizeBatchAdvanced<uint8_t, 3>(
        const NppiSize &maxDst, const vector<NppiImageDescriptor> &pBatchSrc, const vector<NppiImageDescriptor> &pBatchDst, const vector<NppiResizeBatchROI_Advanced> &pBatchROI, NppiInterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
{
    auto src = toCuda(pBatchSrc, nppStreamCtx.hStream);
    auto dst = toCuda(pBatchDst, nppStreamCtx.hStream);
    auto roi = toCuda(pBatchROI, nppStreamCtx.hStream);
    cudaCheck(
        nppiResizeBatch_8u_C3R_Advanced_Ctx(maxDst.width, maxDst.height, src.get(), dst.get(), roi.get(), pBatchSrc.size(), eInterpolation, nppStreamCtx),
        "nppiResizeBatch_8u_C3R_Advanced_Ctx");
}

}
