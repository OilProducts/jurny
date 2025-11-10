#pragma once

#include <volk.h>

namespace render::vkutil {

// Find a memory type index that satisfies the requested property flags.
uint32_t findMemoryType(VkPhysicalDevice phys,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags flags);

// Create a VkBuffer and its backing VkDeviceMemory. Returns false on failure.
bool allocateBuffer(VkDevice device,
                    VkPhysicalDevice phys,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags flags,
                    VkBuffer& outBuf,
                    VkDeviceMemory& outMem);

// Destroy a buffer/memory pair and reset handles to VK_NULL_HANDLE.
void destroyBuffer(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory);

// Create a simple 2D image with an accompanying image view.
bool createImage2D(VkDevice device,
                   VkPhysicalDevice phys,
                   VkExtent2D extent,
                   VkFormat format,
                   VkImageUsageFlags usage,
                   VkImageAspectFlags aspectMask,
                   VkImage& outImage,
                   VkDeviceMemory& outMemory,
                   VkImageView& outView);

// Destroy a VkImage/VkImageView pair and their memory.
void destroyImage(VkDevice device, VkImage& image, VkDeviceMemory& memory, VkImageView& view);

} // namespace render::vkutil
