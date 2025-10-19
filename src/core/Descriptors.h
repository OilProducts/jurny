#pragma once

// Descriptors — set layouts, pools, and bindless indexing helpers.
//
// Provides a thin wrapper over a single descriptor pool with thread-safe allocation.
// Call `init(device, sizes, maxSets)` once, then `allocate(layout)` whenever a
// descriptor set is needed. Individual sets can be freed or the entire pool can
// be reset between frames.

#include <cstdint>
#include <mutex>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace core {

class Descriptors {
public:
    Descriptors() = default;

    bool init(VkDevice device,
              const std::vector<VkDescriptorPoolSize>& poolSizes,
              uint32_t maxSets,
              VkDescriptorPoolCreateFlags flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    void shutdown();

    VkDescriptorSet allocate(VkDescriptorSetLayout layout);
    void free(VkDescriptorSet set);
    void reset();

    [[nodiscard]] bool valid() const { return pool_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkDescriptorPool pool() const { return pool_; }
    [[nodiscard]] VkDevice device() const { return device_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::mutex mutex_;
};

}
