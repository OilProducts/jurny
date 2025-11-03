#include "Upload.h"

#include <algorithm>
#include <cstring>

#include "platform/VulkanContext.h"
#include <spdlog/spdlog.h>

namespace core {

bool UploadContext::init(platform::VulkanContext& vk, VkDeviceSize initialStagingBytes) {
    vk_ = &vk;
    minCapacity_ = std::max<VkDeviceSize>(initialStagingBytes, 256);
    queue_ = vk.transferQueue();
    queueFamily_ = vk.transferFamily();
    if (queue_ == VK_NULL_HANDLE) {
        queue_ = vk.graphicsQueue();
        queueFamily_ = vk.graphicsFamily();
    }

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamily_;
    if (vkCreateCommandPool(vk.device(), &cpci, nullptr, &commandPool_) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to create command pool");
        return false;
    }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = commandPool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(vk.device(), &cbai, &commandBuffer_) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to allocate command buffer");
        return false;
    }

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(vk.device(), &fci, nullptr, &fence_) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to create fence");
        return false;
    }

    if (!ensureStagingCapacity(minCapacity_)) {
        return false;
    }
    return true;
}

bool UploadContext::ensureStagingCapacity(VkDeviceSize bytes) {
    VkDeviceSize required = std::max(bytes, minCapacity_);
    if (stagingBuffer_ && stagingCapacity_ >= required) {
        return true;
    }

    if (stagingBuffer_) {
        vkDestroyBuffer(vk_->device(), stagingBuffer_, nullptr);
        stagingBuffer_ = VK_NULL_HANDLE;
    }
    if (stagingMemory_) {
        vkFreeMemory(vk_->device(), stagingMemory_, nullptr);
        stagingMemory_ = VK_NULL_HANDLE;
    }

    VkDeviceSize newCapacity = stagingCapacity_ == 0 ? required : std::max(required, stagingCapacity_ * 2);

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = newCapacity;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_->device(), &bi, nullptr, &stagingBuffer_) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to create staging buffer ({} bytes)", (uint64_t)newCapacity);
        return false;
    }

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(vk_->device(), stagingBuffer_, &mr);
    uint32_t typeIndex = findMemoryType(mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIndex == UINT32_MAX) {
        spdlog::error("UploadContext: unable to find host-visible memory type for staging buffer");
        return false;
    }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk_->device(), &mai, nullptr, &stagingMemory_) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to allocate staging memory ({} bytes)", (uint64_t)mr.size);
        return false;
    }

    if (vkBindBufferMemory(vk_->device(), stagingBuffer_, stagingMemory_, 0) != VK_SUCCESS) {
        spdlog::error("UploadContext: failed to bind staging memory");
        return false;
    }

    stagingCapacity_ = newCapacity;
    spdlog::info("UploadContext: staging buffer resize -> {} bytes (requested {})",
                 static_cast<uint64_t>(stagingCapacity_), static_cast<uint64_t>(bytes));
    return true;
}

uint32_t UploadContext::findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(vk_->physicalDevice(), &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool UploadContext::uploadBuffer(const void* data, VkDeviceSize bytes, VkBuffer dstBuffer) {
    return uploadBufferRegion(data, bytes, dstBuffer, 0);
}

void UploadContext::flush() {
    if (queue_ != VK_NULL_HANDLE && vk_) {
        vkQueueWaitIdle(queue_);
    }
}

bool UploadContext::uploadBufferRegion(const void* data,
                                       VkDeviceSize bytes,
                                       VkBuffer dstBuffer,
                                       VkDeviceSize dstOffset) {
    if (bytes == 0 || dstBuffer == VK_NULL_HANDLE) {
        return true;
    }
    if (deviceLost_) {
        spdlog::error("UploadContext: device lost, skipping buffer upload ({} bytes)", static_cast<uint64_t>(bytes));
        return false;
    }

    if (!ensureStagingCapacity(bytes)) {
        return false;
    }

    spdlog::info("UploadContext: copy request bytes={} dstBuffer={} dstOffset={} stagingCapacity={}",
                 static_cast<uint64_t>(bytes),
                 static_cast<const void*>(dstBuffer),
                 static_cast<uint64_t>(dstOffset),
                 static_cast<uint64_t>(stagingCapacity_));

    void* mapped = nullptr;
    if (vkMapMemory(vk_->device(), stagingMemory_, 0, bytes, 0, &mapped) != VK_SUCCESS || mapped == nullptr) {
        spdlog::error("UploadContext: failed to map staging memory");
        return false;
    }
    std::memcpy(mapped, data, static_cast<std::size_t>(bytes));
    vkUnmapMemory(vk_->device(), stagingMemory_);

    VkResult waitRes = vkWaitForFences(vk_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    if (waitRes != VK_SUCCESS) {
        spdlog::error("UploadContext: wait for fence failed ({})", static_cast<int>(waitRes));
        deviceLost_ = (waitRes == VK_ERROR_DEVICE_LOST);
        return false;
    }
    vkResetFences(vk_->device(), 1, &fence_);
    spdlog::info("UploadContext: fence cleared before submit (bytes={})", static_cast<uint64_t>(bytes));

    vkResetCommandPool(vk_->device(), commandPool_, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &bi);
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = dstOffset;
    region.size = bytes;
    vkCmdCopyBuffer(commandBuffer_, stagingBuffer_, dstBuffer, 1, &region);
    vkEndCommandBuffer(commandBuffer_);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer_;
    VkResult submitRes = vkQueueSubmit(queue_, 1, &si, fence_);
    if (submitRes != VK_SUCCESS) {
        spdlog::error("UploadContext: vkQueueSubmit failed ({})", static_cast<int>(submitRes));
        deviceLost_ = (submitRes == VK_ERROR_DEVICE_LOST);
        return false;
    }

    spdlog::info("UploadContext: submitted copy bytes={} queueFamily={}", static_cast<uint64_t>(bytes), queueFamily_);

    waitRes = vkWaitForFences(vk_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    if (waitRes != VK_SUCCESS) {
        spdlog::error("UploadContext: wait for fence after submit failed ({})", static_cast<int>(waitRes));
        deviceLost_ = (waitRes == VK_ERROR_DEVICE_LOST);
        return false;
    }
    spdlog::info("UploadContext: copy complete bytes={}", static_cast<uint64_t>(bytes));
    return true;
}

void UploadContext::shutdown() {
    if (!vk_) return;
    flush();
    if (stagingBuffer_) {
        vkDestroyBuffer(vk_->device(), stagingBuffer_, nullptr);
        stagingBuffer_ = VK_NULL_HANDLE;
    }
    if (stagingMemory_) {
        vkFreeMemory(vk_->device(), stagingMemory_, nullptr);
        stagingMemory_ = VK_NULL_HANDLE;
    }
    if (fence_) {
        vkDestroyFence(vk_->device(), fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (commandPool_) {
        vkDestroyCommandPool(vk_->device(), commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }
    commandBuffer_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    stagingCapacity_ = 0;
    vk_ = nullptr;
    deviceLost_ = false;
}

}
