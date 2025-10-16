#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>

// Forward declare Vulkan handles to avoid hard dependency in headers
// Include <volk.h> in the .cpp to define them.
typedef struct VkInstance_T* VkInstance;
typedef struct VkDebugUtilsMessengerEXT_T* VkDebugUtilsMessengerEXT;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef struct VkQueue_T* VkQueue;
typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkPipelineCache_T* VkPipelineCache;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

namespace platform {

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> present; // when a surface is provided
    bool complete() const { return graphics.has_value() && compute.has_value(); }
};

struct DeviceInfo {
    uint32_t apiVersion = 0;
    bool hasTimelineSemaphore = true;
    bool hasDescriptorIndexing = true;
    bool hasBufferDeviceAddress = false;
};

class VulkanContext {
public:
    bool initInstance(const std::vector<const char*>& extraInstanceExts, bool enableValidation = true);
    bool initDevice(VkSurfaceKHR surface = nullptr);
    void shutdown();

    // Accessors
    VkInstance       instance() const { return instance_; }
    VkDebugUtilsMessengerEXT debugMessenger() const { return debugMessenger_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice         device() const { return device_; }
    VkQueue          graphicsQueue() const { return graphicsQueue_; }
    VkQueue          computeQueue() const { return computeQueue_; }
    VkQueue          transferQueue() const { return transferQueue_; }
    uint32_t         graphicsFamily() const { return queueFamilies_.graphics.value(); }
    uint32_t         computeFamily() const { return queueFamilies_.compute.value(); }
    uint32_t         transferFamily() const { return queueFamilies_.transfer.value_or(queueFamilies_.graphics.value()); }
    VkCommandPool    commandPool() const { return commandPool_; }
    VkDescriptorPool descriptorPool() const { return descriptorPool_; }
    VkPipelineCache  pipelineCache() const { return pipelineCache_; }
    const DeviceInfo& deviceInfo() const { return deviceInfo_; }

private:
    bool createInstance(const std::vector<const char*>& extraExts, bool enableValidation);
    void setupDebugMessenger(bool enableValidation);
    bool pickPhysicalDevice(VkSurfaceKHR surface);
    bool createDevice(bool enableValidation, VkSurfaceKHR surface);
    void createCommandPool();
    void createDescriptorPool();
    void createPipelineCache();
    void destroyDebugMessenger();

private:
    VkInstance instance_ = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger_ = nullptr;
    VkPhysicalDevice physicalDevice_ = nullptr;
    VkDevice device_ = nullptr;
    QueueFamilies queueFamilies_{};
    VkQueue graphicsQueue_ = nullptr;
    VkQueue computeQueue_ = nullptr;
    VkQueue transferQueue_ = nullptr;
    VkCommandPool commandPool_ = nullptr;
    VkDescriptorPool descriptorPool_ = nullptr;
    VkPipelineCache pipelineCache_ = nullptr;
    DeviceInfo deviceInfo_{};
    bool validationEnabled_ = false;
};

} // namespace platform
