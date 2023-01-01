#include <cstring>
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

template<> void copy<HeapBuffer, PassiveBuffer>(const void* src, void *dst, size_t size)
{
    std::memcpy(dst, src, size);
}

}
