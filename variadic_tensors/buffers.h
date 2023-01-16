#ifndef BUFFERS_H
#define BUFFERS_H
#include <memory>

namespace vt {

enum class Device
{
    Host,
    Cuda,
    OpenCL
};

class PassiveBuffer {
public:
    constexpr static Device device = Device::Host;
};

template<class T>
concept BufferLike =
        (std::is_constructible_v<T, size_t> && !std::copyable<T>) || std::is_same_v<T, PassiveBuffer>;

template<class T>
concept HostBufferLike =
        BufferLike<T> && T::device == Device::Host;

template<class T>
concept CudaBufferLike =
        BufferLike<T> && T::device == Device::Cuda;

template<typename Derived>
class Buffer
{
public:
    constexpr static Device device = Device::Host;
    Buffer() = delete;
    Buffer(size_t size) : memory(Derived::malloc(size), &Derived::dealloc) {}

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;
    operator PassiveBuffer(){ return {}; }

    void *get() noexcept { return memory.get(); }
    const void *get() const noexcept { return memory.get(); }
private:
    std::unique_ptr<void, void (*)(void *)> memory;
};

class HeapBuffer
{
public:
    constexpr static Device device = Device::Host;
    HeapBuffer(size_t size);

    HeapBuffer(const HeapBuffer &) = delete;
    HeapBuffer(HeapBuffer &&) = delete;
    HeapBuffer &operator=(const HeapBuffer &) = delete;
    HeapBuffer &operator=(HeapBuffer &&) = delete;

    void *get() noexcept;
    const void *get() const noexcept;
private:
    std::unique_ptr<uint8_t[]> memory;
};

}
#endif // BUFFERS_H
