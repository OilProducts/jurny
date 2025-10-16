#include "Swapchain.h"
#include <volk.h>
#include <vector>
#include <algorithm>
#include <cstdio>

namespace platform {

static VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& fmts) {
    for (auto& f : fmts) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
    }
    return fmts.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : fmts[0];
}
static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

bool Swapchain::create(const SwapchainCreateInfo& info) {
    if (!info.device || !info.surface || !info.physicalDevice) return false;
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(info.physicalDevice, info.surface, &caps);
    uint32_t fmtCount=0; vkGetPhysicalDeviceSurfaceFormatsKHR(info.physicalDevice, info.surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount); vkGetPhysicalDeviceSurfaceFormatsKHR(info.physicalDevice, info.surface, &fmtCount, fmts.data());
    uint32_t pmCount=0; vkGetPhysicalDeviceSurfacePresentModesKHR(info.physicalDevice, info.surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> pms(pmCount); vkGetPhysicalDeviceSurfacePresentModesKHR(info.physicalDevice, info.surface, &pmCount, pms.data());
    auto fmt = chooseSurfaceFormat(fmts);
    auto pmode = choosePresentMode(pms);

    VkExtent2D extent{};
    if (caps.currentExtent.width != UINT32_MAX) extent = caps.currentExtent; else {
        extent.width = std::clamp<uint32_t>(info.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height= std::clamp<uint32_t>(info.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    uint32_t imageCount = caps.minImageCount + 1; if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = info.surface;
    sci.minImageCount = imageCount;
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    uint32_t indices[2] = { info.graphicsQueueFamily, info.presentQueueFamily };
    if (info.graphicsQueueFamily != info.presentQueueFamily) {
        sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2; sci.pQueueFamilyIndices = indices;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = pmode;
    sci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(info.device, &sci, nullptr, &swapchain_) != VK_SUCCESS) return false;
    format_ = fmt.format;
    extent_ = extent;

    // Retrieve images and create views
    uint32_t count=0; vkGetSwapchainImagesKHR(info.device, swapchain_, &count, nullptr);
    images_.resize(count); vkGetSwapchainImagesKHR(info.device, swapchain_, &count, images_.data());
    imageViews_ = new VkImageView[count]; imageCount_ = count;
    for (uint32_t i=0;i<count;++i) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt.format;
        vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1; vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(info.device, &vci, nullptr, &imageViews_[i]) != VK_SUCCESS) return false;
    }
    return true;
}

void Swapchain::destroy(VkDevice device) {
    if (imageViews_) {
        for (uint32_t i=0;i<imageCount_;++i) if (imageViews_[i]) vkDestroyImageView(device, imageViews_[i], nullptr);
        delete [] imageViews_; imageViews_ = nullptr; imageCount_ = 0;
    }
    if (swapchain_) { vkDestroySwapchainKHR(device, swapchain_, nullptr); swapchain_ = VK_NULL_HANDLE; }
}

bool Swapchain::resize(const SwapchainCreateInfo& info) {
    destroy(info.device);
    return create(info);
}

} // namespace platform
