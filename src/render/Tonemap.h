#pragma once

#include <volk.h>

namespace render {

// Tonemap encapsulates composite/ACES parameters and dispatch helper.
class Tonemap {
public:
    void setExposure(float value) { exposure_ = value; }
    float exposure() const { return exposure_; }

    void record(VkCommandBuffer cb,
                VkPipeline pipeline,
                VkPipelineLayout layout,
                VkDescriptorSet set,
                uint32_t groupsX,
                uint32_t groupsY,
                VkQueryPool timestamps,
                uint32_t beginQuery,
                uint32_t endQuery) const;

private:
    float exposure_ = 1.0f;
};

} // namespace render
