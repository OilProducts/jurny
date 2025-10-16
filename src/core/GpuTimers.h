#pragma once

// GpuTimers — timestamp queries and pass timing.
namespace core {
class GpuTimers {
public:
    void beginFrame();
    void endFrame();
};
}

