#ifndef BUFFERS_H
#define BUFFERS_H
#include <memory>
#include <cstdlib>

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
concept AllocatedBufferLike =
        requires(T t) {
            { T::malloc(std::declval<size_t>()) } -> std::same_as<void*>;
            { T::dealloc(std::declval<void*>()) } -> std::same_as<void>;
        };

template<class T>
concept BufferLike =
        AllocatedBufferLike<T> || std::is_same_v<T, PassiveBuffer>;

template<class T>
concept HostBufferLike =
        BufferLike<T> && T::device == Device::Host;

template<class T>
concept CudaBufferLike =
        BufferLike<T> && T::device == Device::Cuda;

class Static { Static() = delete; };

class HeapBuffer: Static
{
public:
    constexpr static Device device = Device::Host;
    static void *malloc(size_t size) noexcept { return std::malloc(size); }
    static void dealloc(void *ptr) noexcept { return std::free(ptr); }
};

}
#endif // BUFFERS_H
