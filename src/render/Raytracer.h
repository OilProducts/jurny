#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>
#include "platform/VulkanContext.h"
#include "platform/Swapchain.h"

// Raytracer — compute skeleton (shade sky → composite). M1 scaffolding.
namespace render {

struct GlobalsUBOData {
    float currView[16];
    float currProj[16];
    float prevView[16];
    float prevProj[16];
    float originDeltaPrevToCurr[4];
    float voxelSize, brickSize, Rin, Rout;
    float Rsea, exposure; uint32_t frameIdx, maxBounces;
    uint32_t width, height, raysPerPixel, flags;
    uint32_t worldHashCapacity, worldBrickCount; uint32_t _padA, _padB;
    uint32_t macroHashCapacity, macroDimBricks; float macroSize; uint32_t _padC;
};

class Raytracer {
public:
    bool init(platform::VulkanContext& vk, platform::Swapchain& swap);
    void resize(platform::VulkanContext& vk, platform::Swapchain& swap);
    void updateGlobals(platform::VulkanContext& vk, const GlobalsUBOData& data);
    void record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex);
    void readDebug(platform::VulkanContext& vk, uint32_t frameIdx);
    void shutdown(platform::VulkanContext& vk);

private:
    bool createPipelines(platform::VulkanContext& vk);
    void destroyPipelines(platform::VulkanContext& vk);
    bool createImages(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyImages(platform::VulkanContext& vk);
    bool createDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyDescriptors(platform::VulkanContext& vk);
    bool createWorld(platform::VulkanContext& vk);
    void destroyWorld(platform::VulkanContext& vk);
    bool createQueues(platform::VulkanContext& vk);
    void destroyQueues(platform::VulkanContext& vk);
    void writeQueueHeaders(VkCommandBuffer cb);

private:
    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout      pipeLayout_{};
    VkPipeline            pipeGenerate_{};
    VkPipeline            pipeShade_{};
    VkPipeline            pipeTraverse_{};
    VkPipeline            pipeComposite_{};
    VkDescriptorPool      descPool_{};
    std::vector<VkDescriptorSet> sets_;

    VkBuffer ubo_{}; VkDeviceMemory uboMem_{};
    VkImage accumImage_{}; VkDeviceMemory accumMem_{}; VkImageView accumView_{};
    VkFormat accumFormat_{}; VkExtent2D extent_{};

    // World buffers
    VkBuffer bhBuf_{}; VkDeviceMemory bhMem_{};
    VkBuffer occBuf_{}; VkDeviceMemory occMem_{};
    VkBuffer hkBuf_{}; VkDeviceMemory hkMem_{};
    VkBuffer hvBuf_{}; VkDeviceMemory hvMem_{};
    VkBuffer mkBuf_{}; VkDeviceMemory mkMem_{};
    VkBuffer mvBuf_{}; VkDeviceMemory mvMem_{};
    // Ray/hit/miss queues
    VkBuffer rayQueueBuf_{};      VkDeviceMemory rayQueueMem_{};
    VkBuffer hitQueueBuf_{};      VkDeviceMemory hitQueueMem_{};
    VkBuffer missQueueBuf_{};     VkDeviceMemory missQueueMem_{};
    VkBuffer secondaryQueueBuf_{};VkDeviceMemory secondaryQueueMem_{};
    uint32_t queueCapacity_{};
    uint32_t hashCapacity_{}; uint32_t brickCount_{};
    uint32_t macroCapacity_{}; uint32_t macroDimBricks_{};

    // Debug buffer (GPU→CPU) for per-frame diagnostics
    VkBuffer dbgBuf_{}; VkDeviceMemory dbgMem_{}; // 16*4 bytes sufficient
    uint32_t lastDbgFrame_{}; int lastMcX_{}; int lastMcY_{}; int lastMcZ_{}; int lastPresent_{};
    uint32_t currFrameIdx_{};
};

}
