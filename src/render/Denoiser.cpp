#include "Denoiser.h"

#include <spdlog/spdlog.h>

namespace render {
namespace {
uint32_t findMemoryType(VkPhysicalDevice phys,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}
} // namespace

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
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = { extent.width, extent.height, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk.device(), &ici, nullptr, &target.image) != VK_SUCCESS) {
            spdlog::error("Failed to create denoiser history image ({}x{})", extent.width, extent.height);
            return false;
        }
        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(vk.device(), target.image, &mr);
        uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (typeIndex == UINT32_MAX) {
            spdlog::error("No compatible memory type for denoiser history");
            vkDestroyImage(vk.device(), target.image, nullptr);
            target.image = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = typeIndex;
        if (vkAllocateMemory(vk.device(), &mai, nullptr, &target.memory) != VK_SUCCESS) {
            spdlog::error("Failed to allocate denoiser history memory");
            vkDestroyImage(vk.device(), target.image, nullptr);
            target.image = VK_NULL_HANDLE;
            return false;
        }
        vkBindImageMemory(vk.device(), target.image, target.memory, 0);
        VkImageViewCreateInfo ivci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        ivci.image = target.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk.device(), &ivci, nullptr, &target.view) != VK_SUCCESS) {
            spdlog::error("Failed to create denoiser history view");
            vkFreeMemory(vk.device(), target.memory, nullptr);
            vkDestroyImage(vk.device(), target.image, nullptr);
            target.memory = VK_NULL_HANDLE;
            target.image = VK_NULL_HANDLE;
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
        if (target.view) {
            vkDestroyImageView(vk.device(), target.view, nullptr);
            target.view = VK_NULL_HANDLE;
        }
        if (target.image) {
            vkDestroyImage(vk.device(), target.image, nullptr);
            target.image = VK_NULL_HANDLE;
        }
        if (target.memory) {
            vkFreeMemory(vk.device(), target.memory, nullptr);
            target.memory = VK_NULL_HANDLE;
        }
    };
    for (auto& t : color_) destroyTarget(t);
    for (auto& t : moments_) destroyTarget(t);
}

} // namespace render
