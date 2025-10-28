#pragma once

#include <array>
#include <volk.h>

#include "platform/VulkanContext.h"

namespace render {

// Denoiser manages temporal history targets (color + variance moments).
class Denoiser {
public:
    bool init(platform::VulkanContext& vk, VkExtent2D extent);
    bool resize(platform::VulkanContext& vk, VkExtent2D extent);
    void shutdown(platform::VulkanContext& vk);

    void reset();
    void advance();

    bool initialized() const { return initialized_; }
    uint32_t historyValidFlag() const { return initialized_ ? 1u : 0u; }

    VkImage historyReadImage() const { return color_[readIndex_].image; }
    VkImage historyWriteImage() const { return color_[writeIndex_].image; }
    VkImageView historyReadView() const { return color_[readIndex_].view; }
    VkImageView historyWriteView() const { return color_[writeIndex_].view; }

    VkImage historyMomentsReadImage() const { return moments_[readIndex_].image; }
    VkImage historyMomentsWriteImage() const { return moments_[writeIndex_].image; }
    VkImageView historyMomentsReadView() const { return moments_[readIndex_].view; }
    VkImageView historyMomentsWriteView() const { return moments_[writeIndex_].view; }

private:
    struct HistoryTarget {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    bool createTargets(platform::VulkanContext& vk, VkExtent2D extent);
    void destroyTargets(platform::VulkanContext& vk);

private:
    VkExtent2D extent_{};
    std::array<HistoryTarget, 2> color_{};
    std::array<HistoryTarget, 2> moments_{};
    uint32_t readIndex_ = 0;
    uint32_t writeIndex_ = 1;
    bool initialized_ = false;
};

} // namespace render
