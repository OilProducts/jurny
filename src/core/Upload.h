#pragma once

#include <cstdint>

#include <volk.h>

namespace platform { class VulkanContext; }

// Upload — simple staging helper for buffer uploads.
namespace core {
class UploadContext {
public:
    bool init(platform::VulkanContext& vk, VkDeviceSize initialStagingBytes = 4 * 1024 * 1024);
    bool uploadBuffer(const void* data, VkDeviceSize bytes, VkBuffer dstBuffer);
    void flush();
    void shutdown();

private:
    bool ensureStagingCapacity(VkDeviceSize bytes);
    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;

    platform::VulkanContext* vk_ = nullptr;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkBuffer stagingBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory_ = VK_NULL_HANDLE;
    VkDeviceSize stagingCapacity_ = 0;
    VkDeviceSize minCapacity_ = 0;
};
}
