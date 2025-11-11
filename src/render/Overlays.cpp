#include "Overlays.h"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

#include "render/VulkanUtils.h"

#include "core/Upload.h"

namespace render {

bool Overlays::init(platform::VulkanContext& vk, VkDeviceSize capacityBytes) {
    shutdown(vk);
    if (capacityBytes == 0) {
        return true;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!vkutil::allocateBuffer(vk.device(), vk.physicalDevice(), capacityBytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                buffer,
                                memory)) {
        spdlog::error("Failed to create overlay buffer (size={})", static_cast<uint64_t>(capacityBytes));
        return false;
    }
    buffer_ = buffer;
    memory_ = memory;
    capacity_ = capacityBytes;
    cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
    active_ = false;
    staging_.clear();
    return true;
}

void Overlays::shutdown(platform::VulkanContext& vk) {
    vkutil::destroyBuffer(vk.device(), buffer_, memory_);
    capacity_ = 0;
    cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
    active_ = false;
}

bool Overlays::update(platform::VulkanContext& vk,
                      core::UploadContext& uploadCtx,
                      const std::vector<std::string>& lines,
                      uint32_t maxCols,
                      uint32_t maxRows,
                      uint32_t glyphWidth,
                      uint32_t glyphHeight,
                      uint32_t padX,
                      uint32_t padY) {
    if (!buffer_ || capacity_ == 0) {
        active_ = false;
        return false;
    }

    const uint32_t limitedRows = std::min<uint32_t>(static_cast<uint32_t>(lines.size()), maxRows);
    uint32_t maxWidth = 0;
    for (uint32_t i = 0; i < limitedRows; ++i) {
        maxWidth = std::max<uint32_t>(maxWidth, std::min<uint32_t>(static_cast<uint32_t>(lines[i].size()), maxCols));
    }

    if (maxWidth == 0u || limitedRows == 0u) {
        staging_.assign(4u, 0u);
        uploadCtx.uploadBufferRegion(staging_.data(), sizeof(uint32_t) * staging_.size(), buffer_, 0);
        cols_ = rows_ = pixelWidth_ = pixelHeight_ = 0;
        active_ = false;
        return true;
    }

    const uint32_t stride = maxWidth;
    const uint32_t charCount = stride * limitedRows;
    const VkDeviceSize headerBytes = sizeof(uint32_t) * 4;
    const VkDeviceSize payloadBytes = static_cast<VkDeviceSize>(charCount) * sizeof(uint32_t);
    const VkDeviceSize total = headerBytes + payloadBytes;
    if (total > capacity_) {
        spdlog::warn("Overlay buffer too small (needed {}, capacity {})", static_cast<uint64_t>(total), static_cast<uint64_t>(capacity_));
        active_ = false;
        return false;
    }

    staging_.assign(static_cast<size_t>(headerBytes / sizeof(uint32_t) + charCount), 0u);
    staging_[0] = stride;
    staging_[1] = limitedRows;
    staging_[2] = stride;
    staging_[3] = charCount;

    uint32_t* dst = staging_.data() + 4;
    for (uint32_t row = 0; row < limitedRows; ++row) {
        const std::string& line = lines[row];
        uint32_t len = std::min<uint32_t>(static_cast<uint32_t>(line.size()), stride);
        uint32_t rowOffset = row * stride;
        for (uint32_t col = 0; col < len; ++col) {
            dst[rowOffset + col] = static_cast<uint8_t>(line[col]);
        }
        for (uint32_t col = len; col < stride; ++col) {
            dst[rowOffset + col] = static_cast<uint8_t>(' ');
        }
    }

    if (!uploadCtx.uploadBufferRegion(staging_.data(), total, buffer_, 0)) {
        spdlog::error("Failed to upload overlay data ({} bytes)", static_cast<uint64_t>(total));
        active_ = false;
        return false;
    }

    cols_ = stride;
    rows_ = limitedRows;
    pixelWidth_ = stride * (glyphWidth + padX);
    pixelHeight_ = limitedRows * (glyphHeight + padY);
    active_ = true;
    return true;
}

} // namespace render
