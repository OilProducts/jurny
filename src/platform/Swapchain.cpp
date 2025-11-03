#include "Swapchain.h"
#include <volk.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <spdlog/spdlog.h>

namespace platform {

namespace {

bool supportsStorage(VkPhysicalDevice phys, VkFormat fmt) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(phys, fmt, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

bool selectSurfaceFormat(VkPhysicalDevice phys,
                         const std::vector<VkSurfaceFormatKHR>& fmts,
                         VkSurfaceFormatKHR& chosen) {
    if (fmts.size() == 1 && fmts[0].format == VK_FORMAT_UNDEFINED) {
        const VkColorSpaceKHR cs = fmts[0].colorSpace;
        const VkFormat candidates[] = {
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM
        };
        for (VkFormat fmt : candidates) {
            if (!supportsStorage(phys, fmt)) continue;
            chosen.format = fmt;
            chosen.colorSpace = cs;
            return true;
        }
        return false;
    }

    auto tryPick = [&](VkFormat fmt, VkColorSpaceKHR cs, bool requireColorSpace) -> bool {
        for (auto& f : fmts) {
            if (f.format != fmt) continue;
            if (requireColorSpace && f.colorSpace != cs) continue;
            if (!supportsStorage(phys, f.format)) continue;
            chosen = f;
            return true;
        }
        return false;
    };

    const VkColorSpaceKHR desiredCS = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    // Prefer RGBA so storage image bindings using layout(rgba8) remain valid.
    if (tryPick(VK_FORMAT_R8G8B8A8_UNORM, desiredCS, true)) return true;
    if (tryPick(VK_FORMAT_R8G8B8A8_UNORM, desiredCS, false)) return true;
    if (tryPick(VK_FORMAT_B8G8R8A8_UNORM, desiredCS, true)) return true;
    if (tryPick(VK_FORMAT_B8G8R8A8_UNORM, desiredCS, false)) return true;

    for (auto& f : fmts) {
        if (!supportsStorage(phys, f.format)) continue;
        chosen = f;
        return true;
    }
    return false;
}

static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

} // namespace

bool Swapchain::create(const SwapchainCreateInfo& info) {
    if (!info.device || !info.surface || !info.physicalDevice) return false;
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(info.physicalDevice, info.surface, &caps);
    uint32_t fmtCount=0; vkGetPhysicalDeviceSurfaceFormatsKHR(info.physicalDevice, info.surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    if (fmtCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(info.physicalDevice, info.surface, &fmtCount, fmts.data());
    }
    uint32_t pmCount=0; vkGetPhysicalDeviceSurfacePresentModesKHR(info.physicalDevice, info.surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> pms(pmCount); vkGetPhysicalDeviceSurfacePresentModesKHR(info.physicalDevice, info.surface, &pmCount, pms.data());

    if (fmts.empty()) {
        spdlog::error("No surface formats reported by the driver.");
        return false;
    }
    VkSurfaceFormatKHR fmt{};
    if (!selectSurfaceFormat(info.physicalDevice, fmts, fmt)) {
        spdlog::error("No surface format supports STORAGE usage; cannot build swapchain.");
        return false;
    }
    auto pmode = choosePresentMode(pms);

    VkExtent2D extent{};
    if (caps.currentExtent.width != UINT32_MAX) extent = caps.currentExtent; else {
        extent.width = std::clamp<uint32_t>(info.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height= std::clamp<uint32_t>(info.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    uint32_t imageCount = caps.minImageCount + 1; if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
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
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
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
