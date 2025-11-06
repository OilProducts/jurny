#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
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
    glm::vec4 color;
};

class MeshRenderer {
public:
    bool init(platform::VulkanContext& vk, platform::Swapchain& swap);
    void shutdown(platform::VulkanContext& vk);
    void resize(platform::VulkanContext& vk, platform::Swapchain& swap);

    void setCamera(const glm::mat4& viewProj, const glm::vec3& lightDir);
    void updateInstances(const std::vector<MeshInstance>& instances);
    void record(VkCommandBuffer cb, VkImage targetImage, VkImageView targetView);

    bool ready() const { return ready_; }
    bool hasInstances() const { return instanceCount_ > 0; }

private:
    bool createPipeline(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyPipeline(platform::VulkanContext& vk);
    bool createBuffers(platform::VulkanContext& vk);
    void destroyBuffers(platform::VulkanContext& vk);

    bool ensureInstanceBuffer(size_t count);
    bool createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags flags,
                      VkBuffer& buffer,
                      VkDeviceMemory& memory);
    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags requirements) const;

    struct PushConstants {
        glm::mat4 viewProj{1.0f};
        glm::vec4 lightDir{0.4f, 1.0f, 0.3f, 0.0f};
    };

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_ = VK_NULL_HANDLE;
    uint32_t indexCount_ = 0;

    VkBuffer instanceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory_ = VK_NULL_HANDLE;
    void* instanceMapped_ = nullptr;
    size_t instanceCapacity_ = 0;
    size_t instanceCount_ = 0;

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily_ = 0;
    VkExtent2D swapExtent_{};
    VkFormat colorFormat_ = VK_FORMAT_B8G8R8A8_UNORM;

    PushConstants pushConstants_{};
    bool ready_ = false;
};

} // namespace render
