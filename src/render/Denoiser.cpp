#include "Denoiser.h"

#include <spdlog/spdlog.h>

#include "VulkanUtils.h"

namespace render {

bool Denoiser::init(platform::VulkanContext& vk, VkExtent2D extent) {
    if (!createTargets(vk, extent)) {
        shutdown(vk);
        return false;
    }
    extent_ = extent;
    reset();
    return true;
}

bool Denoiser::resize(platform::VulkanContext& vk, VkExtent2D extent) {
    if (extent.width == extent_.width && extent.height == extent_.height) {
        reset();
        return true;
    }
    shutdown(vk);
    return init(vk, extent);
}

void Denoiser::shutdown(platform::VulkanContext& vk) {
    destroyTargets(vk);
    extent_ = {};
    reset();
}

void Denoiser::reset() {
    readIndex_ = 0;
    writeIndex_ = 1;
    initialized_ = false;
}

void Denoiser::advance() {
    readIndex_ = writeIndex_;
    writeIndex_ ^= 1u;
    initialized_ = true;
}

bool Denoiser::createTargets(platform::VulkanContext& vk, VkExtent2D extent) {
    destroyTargets(vk);

    auto createTarget = [&](HistoryTarget& target) -> bool {
        const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (!vkutil::createImage2D(vk.device(),
                                   vk.physicalDevice(),
                                   extent,
                                   VK_FORMAT_R16G16B16A16_SFLOAT,
                                   usage,
                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                   target.image,
                                   target.memory,
                                   target.view)) {
            spdlog::error("Failed to create denoiser history image ({}x{})", extent.width, extent.height);
            return false;
        }
        return true;
    };

    for (size_t i = 0; i < color_.size(); ++i) {
        if (!createTarget(color_[i])) return false;
        if (!createTarget(moments_[i])) return false;
    }
    return true;
}

void Denoiser::destroyTargets(platform::VulkanContext& vk) {
    auto destroyTarget = [&](HistoryTarget& target) {
        vkutil::destroyImage(vk.device(), target.image, target.memory, target.view);
    };
    for (auto& t : color_) destroyTarget(t);
    for (auto& t : moments_) destroyTarget(t);
}

} // namespace render
