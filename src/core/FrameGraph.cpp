#include "FrameGraph.h"

#include <utility>

namespace core {

void FrameGraph::beginFrame() {
    passes_.clear();
}

void FrameGraph::addPass(std::string name, std::function<void(VkCommandBuffer)> record) {
    passes_.push_back(Pass{std::move(name), std::move(record)});
}

void FrameGraph::execute(VkCommandBuffer commandBuffer) {
    for (auto& pass : passes_) {
        if (pass.record) {
            pass.record(commandBuffer);
        }
    }
}

void FrameGraph::endFrame() {
    passes_.clear();
}

}
