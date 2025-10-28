#pragma once

#include <array>
#include <cstdint>
#include <volk.h>

#include "platform/VulkanContext.h"

namespace render {

// GpuBuffers manages per-frame images (color/motion/albedo/normal/moments),
// ray-processing queues, and traversal statistics buffers.
class GpuBuffers {
public:
    enum class Queue {
        Ray = 0,
        Hit,
        Miss,
        Secondary
    };

    bool init(platform::VulkanContext& vk, VkExtent2D extent);
    bool resize(platform::VulkanContext& vk, VkExtent2D extent);
    void shutdown(platform::VulkanContext& vk);

    VkExtent2D extent() const { return extent_; }

    VkImage colorImage() const { return color_.image; }
    VkImageView colorView() const { return color_.view; }
    VkFormat colorFormat() const { return color_.format; }

    VkImage motionImage() const { return motion_.image; }
    VkImageView motionView() const { return motion_.view; }
    VkFormat motionFormat() const { return motion_.format; }

    VkImage albedoImage() const { return albedo_.image; }
    VkImageView albedoView() const { return albedo_.view; }

    VkImage normalImage() const { return normal_.image; }
    VkImageView normalView() const { return normal_.view; }

    VkImage momentsImage() const { return moments_.image; }
    VkImageView momentsView() const { return moments_.view; }

    VkBuffer queueBuffer(Queue q) const { return queues_[static_cast<size_t>(q)]; }
    VkDeviceMemory queueMemory(Queue q) const { return queueMemory_[static_cast<size_t>(q)]; }
    uint32_t queueCapacity() const { return queueCapacity_; }
    void writeQueueHeaders(VkCommandBuffer cb) const;

    VkBuffer statsBuffer() const { return statsBuf_; }
    VkDeviceMemory statsMemory() const { return statsMem_; }
    void zeroStats(VkCommandBuffer cb) const;

private:
    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    bool createFrameImages(platform::VulkanContext& vk, VkExtent2D extent);
    void destroyFrameImages(platform::VulkanContext& vk);
    bool allocateImage(platform::VulkanContext& vk,
                       VkExtent2D extent,
                       VkFormat format,
                       VkImageUsageFlags usage,
                       ImageResource& out);

    bool createQueueBuffers(platform::VulkanContext& vk, VkExtent2D extent);
    void destroyQueueBuffers(platform::VulkanContext& vk);
    bool createStatsBuffer(platform::VulkanContext& vk);
    void destroyStatsBuffer(platform::VulkanContext& vk);

private:
    VkExtent2D extent_{};
    ImageResource color_{};
    ImageResource motion_{};
    ImageResource albedo_{};
    ImageResource normal_{};
    ImageResource moments_{};

    std::array<VkBuffer, 4> queues_{};
    std::array<VkDeviceMemory, 4> queueMemory_{};
    uint32_t queueCapacity_ = 0;

    VkBuffer statsBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory statsMem_ = VK_NULL_HANDLE;
};

} // namespace render
