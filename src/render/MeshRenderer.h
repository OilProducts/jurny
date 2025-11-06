#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <volk.h>
#include <vector>

namespace platform { class VulkanContext; class Swapchain; }

namespace render {

struct MeshVertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct MeshInstance {
    glm::mat4 model;
    glm::vec3 color;
};

class MeshRenderer {
public:
    bool init(platform::VulkanContext& vk, platform::Swapchain& swap);
    void shutdown(platform::VulkanContext& vk);
    void resize(platform::VulkanContext& vk, platform::Swapchain& swap);

    void updateInstances(const std::vector<MeshInstance>& instances);
    void record(VkCommandBuffer cb, uint32_t swapIndex);

    bool ready() const { return ready_; }

private:
    bool createPipeline(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyPipeline(platform::VulkanContext& vk);
    bool createBuffers(platform::VulkanContext& vk);
    void destroyBuffers(platform::VulkanContext& vk);

    bool ensureInstanceBuffer(size_t count, platform::VulkanContext& vk);

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_ = VK_NULL_HANDLE;
    uint32_t indexCount_ = 0;

    VkBuffer instanceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory_ = VK_NULL_HANDLE;
    size_t instanceCapacity_ = 0;
    size_t instanceCount_ = 0;

    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descSets_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily_ = 0;
    VkExtent2D swapExtent_{};

    bool ready_ = false;
};

} // namespace render

