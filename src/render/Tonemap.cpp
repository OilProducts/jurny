#include "Tonemap.h"

namespace render {

void Tonemap::record(VkCommandBuffer cb,
                     VkPipeline pipeline,
                     VkPipelineLayout layout,
                     VkDescriptorSet set,
                     uint32_t groupsX,
                     uint32_t groupsY,
                     VkQueryPool timestamps,
                     uint32_t beginQuery,
                     uint32_t endQuery) const {
    if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
        return;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (timestamps != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestamps, beginQuery);
    }
    vkCmdDispatch(cb, groupsX, groupsY, 1);
    if (timestamps != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestamps, endQuery);
    }
}

} // namespace render
