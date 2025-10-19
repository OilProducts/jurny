#pragma once

// FrameGraph — describes compute passes and resource lifetimes.
//
// This minimal implementation keeps a linear list of passes per frame. Each pass
// stores a name (for debugging/profiling) and a lambda that records commands on
// the provided command buffer. Future upgrades can extend this with resource
// tracking and barriers.

#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace core {

class FrameGraph {
public:
    struct Pass {
        std::string name;
        std::function<void(VkCommandBuffer)> record;
    };

    void beginFrame();
    void addPass(std::string name, std::function<void(VkCommandBuffer)> record);
    void execute(VkCommandBuffer commandBuffer);
    void endFrame();

    const std::vector<Pass>& passes() const { return passes_; }

private:
    std::vector<Pass> passes_;
};

}
