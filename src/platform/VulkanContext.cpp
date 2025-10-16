#include "VulkanContext.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <string_view>
#include <cstdio>

#include <volk.h>

namespace platform {

static bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name) {
    for (auto& l : layers) if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}
static bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (auto& e : exts) if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData) {
    (void)type; (void)userData;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "[vulkan] %s\n", callbackData->pMessage);
    }
    return VK_FALSE;
}

bool VulkanContext::createInstance(const std::vector<const char*>& extraExts, bool enableValidation) {
    if (volkInitialize() != VK_SUCCESS) return false;

    uint32_t apiVer = 0;
    vkEnumerateInstanceVersion(&apiVer);

    // Instance layers and extensions
    std::vector<const char*> layers;
    std::vector<const char*> exts = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    for (auto* e : extraExts) exts.push_back(e);
    if (enableValidation) {
        uint32_t lc = 0; vkEnumerateInstanceLayerProperties(&lc, nullptr);
        std::vector<VkLayerProperties> avail(lc); vkEnumerateInstanceLayerProperties(&lc, avail.data());
        if (hasLayer(avail, "VK_LAYER_KHRONOS_validation")) layers.push_back("VK_LAYER_KHRONOS_validation");
        uint32_t ec = 0; vkEnumerateInstanceExtensionProperties(nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> instExts(ec); vkEnumerateInstanceExtensionProperties(nullptr, &ec, instExts.data());
        if (hasExtension(instExts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "voxel_app";
    appInfo.apiVersion = apiVer ? apiVer : VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &appInfo;
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    VkResult res = vkCreateInstance(&ci, nullptr, &instance_);
    if (res != VK_SUCCESS) return false;

    volkLoadInstance(instance_);
    return true;
}

void VulkanContext::setupDebugMessenger(bool enableValidation) {
    if (!enableValidation) return;
    VkDebugUtilsMessengerCreateInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    auto pfnCreate = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (pfnCreate) {
        pfnCreate(instance_, &info, nullptr, &debugMessenger_);
    }
}

static bool supportsQueueFlags(VkQueueFamilyProperties2& props2, VkQueueFlags flags) {
    return (props2.queueFamilyProperties.queueFlags & flags) == flags;
}

bool VulkanContext::pickPhysicalDevice(VkSurfaceKHR surface) {
    uint32_t count = 0; vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance_, &count, devs.data());

    // Score devices: prefer discrete, then integrated.
    auto scoreDevice = [&](VkPhysicalDevice pd) -> int {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
        return score;
    };
    std::sort(devs.begin(), devs.end(), [&](auto a, auto b){ return scoreDevice(a) > scoreDevice(b); });

    for (auto pd : devs) {
        // Query queue families
        uint32_t qCount = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qprops.data());
        QueueFamilies qf{};
        for (uint32_t i=0;i<qCount;++i) {
            auto flags = qprops[i].queueFlags;
            if ((flags & VK_QUEUE_GRAPHICS_BIT) && !qf.graphics) qf.graphics = i;
            if ((flags & VK_QUEUE_COMPUTE_BIT) && !qf.compute) qf.compute = i;
            if ((flags & VK_QUEUE_TRANSFER_BIT) && !qf.transfer) qf.transfer = i;
        }
        if (!qf.compute) continue; // require compute at least
        // If a surface is provided, ensure present support exists
        if (surface) {
            bool presentOK = false;
            uint32_t qCount2 = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount2, nullptr);
            std::vector<VkQueueFamilyProperties> qprops2(qCount2);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount2, qprops2.data());
            for (uint32_t i=0;i<qCount2;++i) {
                VkBool32 supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &supported);
                if (supported) { qf.present = i; presentOK = true; break; }
            }
            if (!presentOK) continue;
        }

        physicalDevice_ = pd;
        queueFamilies_ = qf;
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
        deviceInfo_.apiVersion = props.apiVersion;
        return true;
    }
    return false;
}

bool VulkanContext::createDevice(bool enableValidation, VkSurfaceKHR surface) {
    // Desired extensions
    std::vector<const char*> devExts = {
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, // core in 1.2+
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME // core bits in 1.2
    };
    if (surface) devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Query available
    uint32_t ec = 0; vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &ec, nullptr);
    std::vector<VkExtensionProperties> avail(ec);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &ec, avail.data());
    auto keep = [&](const char* name){ return hasExtension(avail, name); };
    devExts.erase(std::remove_if(devExts.begin(), devExts.end(), [&](const char* n){ return !keep(n); }), devExts.end());

    // Features chain
    VkPhysicalDeviceFeatures2 feats2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceVulkan12Features v12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceDescriptorIndexingFeatures descIdx{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
    feats2.pNext = &v12; v12.pNext = &descIdx;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &feats2);

    v12.timelineSemaphore = VK_TRUE;
    descIdx.descriptorBindingPartiallyBound = VK_TRUE;
    descIdx.runtimeDescriptorArray = VK_TRUE;
    descIdx.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE; // illustrative

    deviceInfo_.hasTimelineSemaphore = v12.timelineSemaphore;
    deviceInfo_.hasDescriptorIndexing = descIdx.runtimeDescriptorArray;

    // Queues
    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    std::vector<uint32_t> uniqIdx;
    uniqIdx.push_back(queueFamilies_.graphics.value_or(queueFamilies_.compute.value()));
    uniqIdx.push_back(queueFamilies_.compute.value());
    uniqIdx.push_back(queueFamilies_.transfer.value_or(queueFamilies_.graphics.value_or(queueFamilies_.compute.value())));
    std::sort(uniqIdx.begin(), uniqIdx.end());
    uniqIdx.erase(std::unique(uniqIdx.begin(), uniqIdx.end()), uniqIdx.end());
    for (uint32_t idx : uniqIdx) {
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = idx;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    }

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &feats2;
    dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.empty() ? nullptr : devExts.data();

    VkResult res = vkCreateDevice(physicalDevice_, &dci, nullptr, &device_);
    if (res != VK_SUCCESS) return false;
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queueFamilies_.graphics.value_or(queueFamilies_.compute.value()), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.compute.value(), 0, &computeQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.transfer.value_or(queueFamilies_.graphics.value_or(queueFamilies_.compute.value())), 0, &transferQueue_);

    return true;
}

void VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo ci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    ci.queueFamilyIndex = queueFamilies_.compute.value();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &ci, nullptr, &commandPool_);
}

void VulkanContext::createDescriptorPool() {
    // Big general-purpose pool (numbers are placeholders)
    std::array<VkDescriptorPoolSize, 4> sizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  256 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  256 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024 }
    };
    VkDescriptorPoolCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 2048;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes = sizes.data();
    vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_);
}

void VulkanContext::createPipelineCache() {
    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_);
}

void VulkanContext::destroyDebugMessenger() {
    if (!debugMessenger_) return;
    auto pfnDestroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
    if (pfnDestroy) pfnDestroy(instance_, debugMessenger_, nullptr);
    debugMessenger_ = nullptr;
}

bool VulkanContext::initInstance(const std::vector<const char*>& extraExts, bool enableValidation) {
    validationEnabled_ = enableValidation;
    if (!createInstance(extraExts, enableValidation)) return false;
    setupDebugMessenger(enableValidation);
    return true;
}

bool VulkanContext::initDevice(VkSurfaceKHR surface) {
    if (!pickPhysicalDevice(surface)) return false;
    if (!createDevice(validationEnabled_, surface)) return false;
    createCommandPool();
    createDescriptorPool();
    createPipelineCache();
    return true;
}

void VulkanContext::shutdown() {
    if (device_) {
        if (pipelineCache_) { vkDestroyPipelineCache(device_, pipelineCache_, nullptr); pipelineCache_ = nullptr; }
        if (descriptorPool_) { vkDestroyDescriptorPool(device_, descriptorPool_, nullptr); descriptorPool_ = nullptr; }
        if (commandPool_) { vkDestroyCommandPool(device_, commandPool_, nullptr); commandPool_ = nullptr; }
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr); device_ = nullptr;
    }
    destroyDebugMessenger();
    if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = nullptr; }
}

} // namespace platform
