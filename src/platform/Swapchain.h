#pragma once

#include <cstdint>
typedef struct VkDevice_T* VkDevice;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;
typedef struct VkSwapchainKHR_T* VkSwapchainKHR;
typedef struct VkImageView_T* VkImageView;
typedef struct VkSemaphore_T* VkSemaphore;
typedef struct VkFence_T* VkFence;
typedef struct VkQueue_T* VkQueue;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef uint64_t VkDeviceSize;

namespace platform {

struct SwapchainCreateInfo {
    VkDevice device{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physicalDevice{};
    uint32_t graphicsQueueFamily = 0;
    uint32_t presentQueueFamily = 0;
    int width = 0;
    int height = 0;
};

class Swapchain {
public:
    bool create(const SwapchainCreateInfo&);
    void destroy(VkDevice device);
    bool resize(const SwapchainCreateInfo& info);

    VkSwapchainKHR handle() const { return swapchain_; }
    VkImageView* imageViews() { return imageViews_; }
    uint32_t imageCount() const { return imageCount_; }

private:
    VkSwapchainKHR swapchain_{};
    VkImageView* imageViews_{};
    uint32_t imageCount_{};
};

} // namespace platform
