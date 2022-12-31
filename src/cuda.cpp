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

}
