#pragma once

// GpuAllocator — VMA wrapper for buffers/images.
namespace core {
class GpuAllocator {
public:
    void init();
    void shutdown();
};
}

