#pragma once

// GpuBuffers — ray/hit queues and frame images.
namespace render {
class GpuBuffers {
public:
    void create(int width, int height);
    void destroy();
};
}

