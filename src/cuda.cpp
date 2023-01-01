#include <cuda_runtime.h>

#include "./cuda.h"


namespace vt {

void *CudaBuffer::malloc(size_t size)
{
    void *ptr;
    auto result = cudaMalloc(&ptr, size);
    if(result != cudaSuccess)
        throw std::runtime_error("cudaMalloc error");
    return ptr;
}

void CudaBuffer::dealloc(void *ptr)
{
    auto result = cudaFree(ptr);
    if(result != cudaSuccess)
        throw std::runtime_error("cudaFree error");
}

std::ostream &operator<<(std::ostream &stream, const Empty &t) noexcept
{
    return stream << "Empty";
}

void *PinnedBuffer::malloc(size_t size)
{
    void *ptr;
    auto result = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if(result != cudaSuccess)
        throw std::runtime_error("cudaHostAlloc error");
    return ptr;
}

void PinnedBuffer::dealloc(void *ptr)
{
    auto result = cudaFreeHost(ptr);
    if(result != cudaSuccess)
        throw std::runtime_error("cudaFree error");
}

template<> void copy<PassiveBuffer, CudaBuffer>(const void* src, void *dst, size_t size)
{
    auto result = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
        throw std::runtime_error("cudaMemcpy error");
}

template<> void copy<CudaBuffer, PassiveBuffer>(const void* src, void *dst, size_t size)
{
    auto result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
        throw std::runtime_error("cudaMemcpy error");
}

template<> void copy<CudaBuffer, HeapBuffer>(const void* src, void *dst, size_t size)
{
    copy<CudaBuffer, PassiveBuffer>(src, dst, size);
}

}
