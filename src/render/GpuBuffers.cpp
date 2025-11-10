#include "GpuBuffers.h"

#include <algorithm>
#include <array>
#include <cstring>

#include <spdlog/spdlog.h>

#include "VulkanUtils.h"

namespace render {
namespace {

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
        vkutil::destroyImage(vk.device(), img.image, img.memory, img.view);
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
    if (!vkutil::createImage2D(vk.device(),
                               vk.physicalDevice(),
                               extent,
                               format,
                               usage,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               out.image,
                               out.memory,
                               out.view)) {
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
    constexpr VkDeviceSize kRayPayload = 128;
    constexpr VkDeviceSize kHitPayload = 160;
    constexpr VkDeviceSize kMissPayload = 128;
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
        if (!vkutil::allocateBuffer(vk.device(),
                                    vk.physicalDevice(),
                                    size,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    buf,
                                    mem)) {
            return false;
        }
    }
    return true;
}

void GpuBuffers::destroyQueueBuffers(platform::VulkanContext& vk) {
    for (size_t i = 0; i < queues_.size(); ++i) {
        vkutil::destroyBuffer(vk.device(), queues_[i], queueMemory_[i]);
    }
    queueCapacity_ = 0;
}

bool GpuBuffers::createStatsBuffer(platform::VulkanContext& vk) {
    destroyStatsBuffer(vk);
    return vkutil::allocateBuffer(vk.device(),
                                  vk.physicalDevice(),
                                  sizeof(uint32_t) * 8,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  statsBuf_,
                                  statsMem_);
}

void GpuBuffers::destroyStatsBuffer(platform::VulkanContext& vk) {
    vkutil::destroyBuffer(vk.device(), statsBuf_, statsMem_);
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
