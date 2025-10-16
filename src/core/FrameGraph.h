#pragma once

// FrameGraph — describes compute passes and resource lifetimes.
namespace core {
class FrameGraph {
public:
    void beginFrame();
    void endFrame();
};
}

