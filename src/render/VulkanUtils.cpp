#include "VulkanUtils.h"

#include <spdlog/spdlog.h>

namespace render::vkutil {

uint32_t findMemoryType(VkPhysicalDevice phys,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const bool supported = (typeBits & (1u << i)) != 0;
        const bool matchesFlags = (props.memoryTypes[i].propertyFlags & flags) == flags;
        if (supported && matchesFlags) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool allocateBuffer(VkDevice device,
                    VkPhysicalDevice phys,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags flags,
                    VkBuffer& outBuf,
                    VkDeviceMemory& outMem) {
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = size;
    createInfo.usage = usage;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &createInfo, nullptr, &outBuf) != VK_SUCCESS) {
        spdlog::error("vkCreateBuffer failed (size={} usage=0x{:x})",
                      static_cast<unsigned long long>(size),
                      static_cast<unsigned int>(usage));
        outBuf = VK_NULL_HANDLE;
        outMem = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, outBuf, &requirements);
    uint32_t typeIndex = findMemoryType(phys, requirements.memoryTypeBits, flags);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No compatible memory type for buffer allocation (flags=0x{:x})",
                      static_cast<unsigned int>(flags));
        vkDestroyBuffer(device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        outMem = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMem) != VK_SUCCESS) {
        spdlog::error("vkAllocateMemory failed (size={} type={})",
                      static_cast<unsigned long long>(requirements.size),
                      typeIndex);
        vkDestroyBuffer(device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        outMem = VK_NULL_HANDLE;
        return false;
    }

    vkBindBufferMemory(device, outBuf, outMem, 0);
    return true;
}

void destroyBuffer(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory) {
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

bool createImage2D(VkDevice device,
                   VkPhysicalDevice phys,
                   VkExtent2D extent,
                   VkFormat format,
                   VkImageUsageFlags usage,
                   VkImageAspectFlags aspectMask,
                   VkImage& outImage,
                   VkDeviceMemory& outMemory,
                   VkImageView& outView) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = { extent.width, extent.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
        spdlog::error("vkCreateImage failed ({}x{}, fmt={})",
                      extent.width,
                      extent.height,
                      static_cast<int>(format));
        outImage = VK_NULL_HANDLE;
        outMemory = VK_NULL_HANDLE;
        outView = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(device, outImage, &mr);
    uint32_t typeIndex = findMemoryType(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No compatible memory type for image (fmt={})", static_cast<int>(format));
        vkDestroyImage(device, outImage, nullptr);
        outImage = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = mr.size;
    allocInfo.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
        spdlog::error("vkAllocateMemory failed for image ({} bytes)", static_cast<unsigned long long>(mr.size));
        vkDestroyImage(device, outImage, nullptr);
        outImage = VK_NULL_HANDLE;
        outMemory = VK_NULL_HANDLE;
        return false;
    }

    vkBindImageMemory(device, outImage, outMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.image = outImage;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectMask;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS) {
        spdlog::error("vkCreateImageView failed");
        vkFreeMemory(device, outMemory, nullptr);
        vkDestroyImage(device, outImage, nullptr);
        outMemory = VK_NULL_HANDLE;
        outImage = VK_NULL_HANDLE;
        outView = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void destroyImage(VkDevice device, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE) {
        vkDestroyImage(device, image, nullptr);
        image = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

} // namespace render::vkutil
