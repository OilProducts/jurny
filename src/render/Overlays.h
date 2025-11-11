#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <volk.h>

#include "platform/VulkanContext.h"

namespace core { class UploadContext; }

namespace render {

// Overlays uploads HUD text to a storage buffer consumed by the overlay shader.
class Overlays {
public:
    bool init(platform::VulkanContext& vk, VkDeviceSize capacityBytes);
    void shutdown(platform::VulkanContext& vk);

    bool update(platform::VulkanContext& vk,
                core::UploadContext& uploadCtx,
                const std::vector<std::string>& lines,
                uint32_t maxCols,
                uint32_t maxRows,
                uint32_t glyphWidth,
                uint32_t glyphHeight,
                uint32_t padX,
                uint32_t padY);

    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize capacity() const { return capacity_; }

    uint32_t cols() const { return cols_; }
    uint32_t rows() const { return rows_; }
    uint32_t pixelWidth() const { return pixelWidth_; }
    uint32_t pixelHeight() const { return pixelHeight_; }
    bool active() const { return active_; }

private:
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize capacity_ = 0;
    uint32_t cols_ = 0;
    uint32_t rows_ = 0;
    uint32_t pixelWidth_ = 0;
    uint32_t pixelHeight_ = 0;
    bool active_ = false;
    std::vector<uint32_t> staging_;
};

} // namespace render
