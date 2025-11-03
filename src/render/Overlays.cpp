#include "Overlays.h"

#include <algorithm>
#include <cstring>

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

bool Overlays::init(platform::VulkanContext& vk, VkDeviceSize capacityBytes) {
    shutdown(vk);
    if (capacityBytes == 0) {
        return true;
    }
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = capacityBytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk.device(), &bi, nullptr, &buffer_) != VK_SUCCESS) {
        spdlog::error("Failed to create overlay buffer (size={})", capacityBytes);
        buffer_ = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(vk.device(), buffer_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No host-visible memory type for overlay buffer");
        vkDestroyBuffer(vk.device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = std::max<VkDeviceSize>(mr.size, capacityBytes);
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &memory_) != VK_SUCCESS) {
        spdlog::error("Failed to allocate overlay memory");
        vkDestroyBuffer(vk.device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(vk.device(), buffer_, memory_, 0);
    capacity_ = capacityBytes;
    cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
    active_ = false;
    return true;
}

void Overlays::shutdown(platform::VulkanContext& vk) {
    if (buffer_) {
        vkDestroyBuffer(vk.device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_) {
        vkFreeMemory(vk.device(), memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    capacity_ = 0;
    cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
    active_ = false;
}

bool Overlays::update(platform::VulkanContext& vk,
                      const std::vector<std::string>& lines,
                      uint32_t maxCols,
                      uint32_t maxRows,
                      uint32_t glyphWidth,
                      uint32_t glyphHeight,
                      uint32_t padX,
                      uint32_t padY) {
    if (!buffer_ || !memory_) {
        active_ = false;
        return false;
    }

    const uint32_t limitedRows = std::min<uint32_t>(static_cast<uint32_t>(lines.size()), maxRows);
    uint32_t maxWidth = 0;
    for (uint32_t i = 0; i < limitedRows; ++i) {
        maxWidth = std::max<uint32_t>(maxWidth, std::min<uint32_t>(static_cast<uint32_t>(lines[i].size()), maxCols));
    }
    if (maxWidth == 0u || limitedRows == 0u) {
        if (buffer_ && memory_) {
            void* ptr = nullptr;
            if (vkMapMemory(vk.device(), memory_, 0, capacity_, 0, &ptr) == VK_SUCCESS && ptr) {
                std::memset(ptr, 0, static_cast<size_t>(capacity_));
                vkUnmapMemory(vk.device(), memory_);
            }
        }
        cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
        active_ = false;
        return false;
    }

    const uint32_t stride = maxWidth;
    const uint32_t charCount = stride * limitedRows;
    const VkDeviceSize headerBytes = sizeof(uint32_t) * 4;
    const VkDeviceSize payloadBytes = static_cast<VkDeviceSize>(charCount) * sizeof(uint32_t);
    const VkDeviceSize total = headerBytes + payloadBytes;
    if (total > capacity_) {
        spdlog::warn("Overlay buffer too small (needed {}, capacity {})", total, capacity_);
        active_ = false;
        return false;
    }

    uint32_t* mapped = nullptr;
    if (vkMapMemory(vk.device(), memory_, 0, total, 0, reinterpret_cast<void**>(&mapped)) != VK_SUCCESS || !mapped) {
        spdlog::error("Failed to map overlay buffer");
        active_ = false;
        return false;
    }

    std::memset(mapped, 0, static_cast<size_t>(total));
    mapped[0] = stride;
    mapped[1] = limitedRows;
    mapped[2] = stride;
    mapped[3] = charCount;

    uint32_t* dst = mapped + 4;
    for (uint32_t row = 0; row < limitedRows; ++row) {
        const std::string& line = lines[row];
        uint32_t len = std::min<uint32_t>(static_cast<uint32_t>(line.size()), stride);
        for (uint32_t col = 0; col < len; ++col) {
            dst[row * stride + col] = static_cast<unsigned char>(line[col]);
        }
        for (uint32_t col = len; col < stride; ++col) {
            dst[row * stride + col] = static_cast<unsigned char>(' ');
        }
    }

    vkUnmapMemory(vk.device(), memory_);

    cols_ = stride;
    rows_ = limitedRows;
    pixelWidth_ = stride * (glyphWidth + padX);
    pixelHeight_ = limitedRows * (glyphHeight + padY);
    active_ = (rows_ > 0 && cols_ > 0);
    return true;
}

} // namespace render
