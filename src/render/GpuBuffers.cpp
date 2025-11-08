#include "GpuBuffers.h"

#include <algorithm>
#include <array>
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

bool allocateBuffer(VkDevice device,
                    VkPhysicalDevice phys,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags flags,
                    VkBuffer& outBuf,
                    VkDeviceMemory& outMem) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &outBuf) != VK_SUCCESS) {
        spdlog::error("Failed to create buffer (size={})", size);
        return false;
    }
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device, outBuf, &mr);
    uint32_t typeIndex = findMemoryType(phys, mr.memoryTypeBits, flags);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No compatible memory type for buffer allocation");
        vkDestroyBuffer(device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(device, &mai, nullptr, &outMem) != VK_SUCCESS) {
        spdlog::error("Failed to allocate buffer memory (size={})", mr.size);
        vkDestroyBuffer(device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device, outBuf, outMem, 0);
    return true;
}

uint32_t nextPow2(uint32_t v) {
    if (v <= 1u) return 1u;
    v -= 1u;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}
} // namespace

bool GpuBuffers::init(platform::VulkanContext& vk, VkExtent2D extent) {
    if (!createFrameImages(vk, extent)) {
        shutdown(vk);
        return false;
    }
    if (!createQueueBuffers(vk, extent)) {
        shutdown(vk);
        return false;
    }
    if (!createStatsBuffer(vk)) {
        shutdown(vk);
        return false;
    }
    extent_ = extent;
    return true;
}

bool GpuBuffers::resize(platform::VulkanContext& vk, VkExtent2D extent) {
    if (extent.width == extent_.width && extent.height == extent_.height) {
        return true;
    }
    shutdown(vk);
    return init(vk, extent);
}

void GpuBuffers::shutdown(platform::VulkanContext& vk) {
    destroyStatsBuffer(vk);
    destroyQueueBuffers(vk);
    destroyFrameImages(vk);
    extent_ = {};
}

bool GpuBuffers::createFrameImages(platform::VulkanContext& vk, VkExtent2D extent) {
    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!allocateImage(vk, extent, VK_FORMAT_R16G16B16A16_SFLOAT, usage, color_)) return false;
    if (!allocateImage(vk, extent, VK_FORMAT_R16G16B16A16_SFLOAT, usage, motion_)) return false;
    if (!allocateImage(vk, extent, VK_FORMAT_R16G16B16A16_SFLOAT, usage, albedo_)) return false;
    if (!allocateImage(vk, extent, VK_FORMAT_R16G16B16A16_SFLOAT, usage, normal_)) return false;
    if (!allocateImage(vk, extent, VK_FORMAT_R16G16B16A16_SFLOAT, usage, moments_)) return false;
    return true;
}

void GpuBuffers::destroyFrameImages(platform::VulkanContext& vk) {
    auto destroy = [&](ImageResource& img) {
        if (img.view) { vkDestroyImageView(vk.device(), img.view, nullptr); img.view = VK_NULL_HANDLE; }
        if (img.image) { vkDestroyImage(vk.device(), img.image, nullptr); img.image = VK_NULL_HANDLE; }
        if (img.memory) { vkFreeMemory(vk.device(), img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
        img.format = VK_FORMAT_UNDEFINED;
    };
    destroy(color_);
    destroy(motion_);
    destroy(albedo_);
    destroy(normal_);
    destroy(moments_);
}

bool GpuBuffers::allocateImage(platform::VulkanContext& vk,
                               VkExtent2D extent,
                               VkFormat format,
                               VkImageUsageFlags usage,
                               ImageResource& out) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = { extent.width, extent.height, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(vk.device(), &ici, nullptr, &out.image) != VK_SUCCESS) {
        spdlog::error("Failed to create GPU image ({}x{}, fmt={})", extent.width, extent.height, static_cast<int>(format));
        return false;
    }

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(vk.device(), out.image, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No compatible memory type for image allocation");
        vkDestroyImage(vk.device(), out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &out.memory) != VK_SUCCESS) {
        spdlog::error("Failed to allocate image memory (size={})", mr.size);
        vkDestroyImage(vk.device(), out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(vk.device(), out.image, out.memory, 0);

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = out.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = format;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(vk.device(), &ivci, nullptr, &out.view) != VK_SUCCESS) {
        spdlog::error("Failed to create image view");
        vkFreeMemory(vk.device(), out.memory, nullptr);
        vkDestroyImage(vk.device(), out.image, nullptr);
        out.memory = VK_NULL_HANDLE;
        out.image = VK_NULL_HANDLE;
        return false;
    }
    out.format = format;
    return true;
}

bool GpuBuffers::createQueueBuffers(platform::VulkanContext& vk, VkExtent2D extent) {
    destroyQueueBuffers(vk);
    uint64_t pixelCount = static_cast<uint64_t>(extent.width) * static_cast<uint64_t>(extent.height);
    queueCapacity_ = nextPow2(static_cast<uint32_t>(std::max<uint64_t>(1ull, pixelCount)));

    constexpr VkDeviceSize kHeaderBytes = sizeof(uint32_t) * 4;
    constexpr VkDeviceSize kRayPayload = 80;
    constexpr VkDeviceSize kHitPayload = 96;
    constexpr VkDeviceSize kMissPayload = 80;
    const std::array<VkDeviceSize, 4> payloadSizes = {
        kRayPayload,
        kHitPayload,
        kMissPayload,
        kRayPayload
    };

    for (size_t i = 0; i < queues_.size(); ++i) {
        VkBuffer& buf = queues_[i];
        VkDeviceMemory& mem = queueMemory_[i];
        VkDeviceSize size = kHeaderBytes + payloadSizes[i] * queueCapacity_;
        if (!allocateBuffer(vk.device(), vk.physicalDevice(), size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            buf, mem)) {
            return false;
        }
    }
    return true;
}

void GpuBuffers::destroyQueueBuffers(platform::VulkanContext& vk) {
    for (size_t i = 0; i < queues_.size(); ++i) {
        if (queues_[i]) {
            vkDestroyBuffer(vk.device(), queues_[i], nullptr);
            queues_[i] = VK_NULL_HANDLE;
        }
        if (queueMemory_[i]) {
            vkFreeMemory(vk.device(), queueMemory_[i], nullptr);
            queueMemory_[i] = VK_NULL_HANDLE;
        }
    }
    queueCapacity_ = 0;
}

bool GpuBuffers::createStatsBuffer(platform::VulkanContext& vk) {
    destroyStatsBuffer(vk);
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = sizeof(uint32_t) * 8;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk.device(), &bi, nullptr, &statsBuf_) != VK_SUCCESS) {
        spdlog::error("Failed to create stats buffer");
        return false;
    }
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(vk.device(), statsBuf_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("No compatible memory type for stats buffer");
        vkDestroyBuffer(vk.device(), statsBuf_, nullptr);
        statsBuf_ = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &statsMem_) != VK_SUCCESS) {
        spdlog::error("Failed to allocate stats buffer memory");
        vkDestroyBuffer(vk.device(), statsBuf_, nullptr);
        statsBuf_ = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(vk.device(), statsBuf_, statsMem_, 0);
    return true;
}

void GpuBuffers::destroyStatsBuffer(platform::VulkanContext& vk) {
    if (statsBuf_) {
        vkDestroyBuffer(vk.device(), statsBuf_, nullptr);
        statsBuf_ = VK_NULL_HANDLE;
    }
    if (statsMem_) {
        vkFreeMemory(vk.device(), statsMem_, nullptr);
        statsMem_ = VK_NULL_HANDLE;
    }
}

void GpuBuffers::writeQueueHeaders(VkCommandBuffer cb) const {
    if (queueCapacity_ == 0) return;
    struct Header {
        uint32_t head;
        uint32_t tail;
        uint32_t capacity;
        uint32_t dropped;
    } hdr{0u, 0u, queueCapacity_, 0u};
    for (VkBuffer buf : queues_) {
        if (buf) {
            vkCmdUpdateBuffer(cb, buf, 0, sizeof(Header), &hdr);
        }
    }
}

void GpuBuffers::resetQueueHeader(VkCommandBuffer cb, Queue q) const {
    if (queueCapacity_ == 0) return;
    VkBuffer buf = queues_[static_cast<size_t>(q)];
    if (!buf) return;
    struct Header {
        uint32_t head;
        uint32_t tail;
        uint32_t capacity;
        uint32_t dropped;
    } hdr{0u, 0u, queueCapacity_, 0u};
    vkCmdUpdateBuffer(cb, buf, 0, sizeof(Header), &hdr);
}

void GpuBuffers::zeroStats(VkCommandBuffer cb) const {
    if (!statsBuf_) return;
    vkCmdFillBuffer(cb, statsBuf_, 0, sizeof(uint32_t) * 8, 0);
}

} // namespace render
