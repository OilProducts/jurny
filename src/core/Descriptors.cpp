#include "Descriptors.h"

#include <cassert>

namespace core {

bool Descriptors::init(VkDevice device,
                       const std::vector<VkDescriptorPoolSize>& poolSizes,
                       uint32_t maxSets,
                       VkDescriptorPoolCreateFlags flags) {
    shutdown();
    if (!device || poolSizes.empty() || maxSets == 0u) return false;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags = flags;
    info.maxSets = maxSets;
    info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    info.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &info, nullptr, &pool_) != VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        return false;
    }

    device_ = device;
    return true;
}

void Descriptors::shutdown() {
    if (pool_ && device_) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
    pool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

VkDescriptorSet Descriptors::allocate(VkDescriptorSetLayout layout) {
    if (!layout || !pool_) return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> lock(mutex_);
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool_;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device_, &info, &set) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return set;
}

void Descriptors::free(VkDescriptorSet set) {
    if (!set || !pool_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    vkFreeDescriptorSets(device_, pool_, 1, &set);
}

void Descriptors::reset() {
    if (pool_) {
        vkResetDescriptorPool(device_, pool_, 0);
    }
}

}
