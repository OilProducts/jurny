#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <glm/vec3.hpp>

#include "core/Upload.h"
#include "platform/VulkanContext.h"
#include "platform/Swapchain.h"
#include "world/BrickStore.h"

// Raytracer — compute skeleton (shade sky → composite). M1 scaffolding.
namespace core { class AssetRegistry; }

namespace render {

struct GlobalsUBOData {
    float currView[16];
    float currProj[16];
    float prevView[16];
    float prevProj[16];
    float renderOrigin[4];
    float originDeltaPrevToCurr[4];
    float voxelSize, brickSize, Rin, Rout;
    float Rsea, planetRadius, exposure, _padExposure;
    uint32_t frameIdx, maxBounces, width, height;
    uint32_t raysPerPixel, flags, worldHashCapacity, worldBrickCount;
    uint32_t macroHashCapacity, macroDimBricks;
    float macroSize;
    uint32_t historyValid;
    float noiseContinentFreq, noiseContinentAmp, noiseDetailFreq, noiseDetailAmp;
    float noiseWarpFreq, noiseWarpAmp, noiseCaveFreq, noiseCaveAmp;
    float noiseCaveThreshold, noiseMinHeight, noiseMaxHeight, noisePad2;
    uint32_t noiseSeed, noiseContinentOctaves, noiseDetailOctaves, noiseCaveOctaves;
};
static_assert(sizeof(GlobalsUBOData) % 16 == 0, "GlobalsUBOData must align to 16 bytes");

class Raytracer {
public:
    bool init(platform::VulkanContext& vk, platform::Swapchain& swap);
    void resize(platform::VulkanContext& vk, platform::Swapchain& swap);
    void updateGlobals(platform::VulkanContext& vk, const GlobalsUBOData& data);
    void record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex);
    void recordOverlay(VkCommandBuffer cb, uint32_t swapIndex);
    void readDebug(platform::VulkanContext& vk, uint32_t frameIdx);
    void shutdown(platform::VulkanContext& vk);

    void setAssetRegistry(const core::AssetRegistry* assets) { assets_ = assets; }

    const world::BrickStore* worldStore() const { return brickStore_.get(); }
    bool addRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord, world::CpuWorld&& cpu);
    bool removeRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord);
    uint32_t brickCount() const { return brickCount_; }
    size_t residentRegionCount() const { return regionWorlds_.size(); }
    std::array<double, 4> gpuTimingsMs() const;
    void updateOverlayHUD(platform::VulkanContext& vk, const std::vector<std::string>& lines);

private:
    bool createPipelines(platform::VulkanContext& vk);
    void destroyPipelines(platform::VulkanContext& vk);
    bool createImages(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyImages(platform::VulkanContext& vk);
    bool createDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap);
    void destroyDescriptors(platform::VulkanContext& vk);
    bool createWorld(platform::VulkanContext& vk);
    void destroyWorld(platform::VulkanContext& vk);
    struct BufferResource {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize capacity = 0;
        VkDeviceSize size = 0;
        VkBufferUsageFlags usage = 0;
    };
    bool ensureBuffer(platform::VulkanContext& vk, BufferResource& buf, VkDeviceSize requiredBytes, VkBufferUsageFlags usage);
    void destroyBuffer(platform::VulkanContext& vk, BufferResource& buf);
    void markWorldDescriptorsDirty();
    void refreshWorldDescriptors(platform::VulkanContext& vk);
    bool createQueues(platform::VulkanContext& vk);
    void destroyQueues(platform::VulkanContext& vk);
    void writeQueueHeaders(VkCommandBuffer cb);
    bool createProfilingResources(platform::VulkanContext& vk);
   void destroyProfilingResources(platform::VulkanContext& vk);
   bool createStatsBuffer(platform::VulkanContext& vk);
   void destroyStatsBuffer(platform::VulkanContext& vk);
    void updateFrameDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap, uint32_t swapIndex);

private:
    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout      pipeLayout_{};
    VkPipeline            pipeGenerate_{};
    VkPipeline            pipeShade_{};
    VkPipeline            pipeTemporal_{};
    VkPipeline            pipeTraverse_{};
    VkPipeline            pipeComposite_{};
    VkPipeline            pipeOverlay_{};
    VkDescriptorPool      descPool_{};
    std::vector<VkDescriptorSet> sets_;

    VkBuffer ubo_{}; VkDeviceMemory uboMem_{};
    VkImage currColorImage_{}; VkDeviceMemory currColorMem_{}; VkImageView currColorView_{};
    VkFormat currColorFormat_{}; VkExtent2D extent_{};
    VkImage motionImage_{}; VkDeviceMemory motionMem_{}; VkImageView motionView_{};
    VkFormat motionFormat_{};
    VkImage albedoImage_{}; VkDeviceMemory albedoMem_{}; VkImageView albedoView_{};
    VkImage normalImage_{}; VkDeviceMemory normalMem_{}; VkImageView normalView_{};
    VkImage momentsImage_{}; VkDeviceMemory momentsMem_{}; VkImageView momentsView_{};
    struct HistoryImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::array<HistoryImage, 2> history_{};
    std::array<HistoryImage, 2> historyMoments_{};
    uint32_t historyReadIndex_ = 0;
    uint32_t historyWriteIndex_ = 1;
    bool historyInitialized_ = false;

    const core::AssetRegistry* assets_ = nullptr;

    // World buffers
    BufferResource bhBuf_{};
    BufferResource occBuf_{};
    BufferResource hkBuf_{};
    BufferResource hvBuf_{};
    BufferResource mkBuf_{};
    BufferResource mvBuf_{};
    BufferResource paletteBuf_{};
    BufferResource matIdxBuf_{};
    BufferResource materialTableBuf_{};
    // Ray/hit/miss queues
    VkBuffer rayQueueBuf_{};      VkDeviceMemory rayQueueMem_{};
    VkBuffer hitQueueBuf_{};      VkDeviceMemory hitQueueMem_{};
    VkBuffer missQueueBuf_{};     VkDeviceMemory missQueueMem_{};
    VkBuffer secondaryQueueBuf_{};VkDeviceMemory secondaryQueueMem_{};
    uint32_t queueCapacity_{};
    VkBuffer statsBuf_{}; VkDeviceMemory statsMem_{};
    struct TraversalStatsHost {
        uint32_t macroVisited;
        uint32_t macroSkipped;
        uint32_t brickSteps;
        uint32_t microSteps;
        uint32_t hitsTotal;
        uint32_t pad0;
        uint32_t pad1;
        uint32_t pad2;
    } statsHost_{};
    uint32_t hashCapacity_{}; uint32_t brickCount_{};
    uint32_t macroCapacity_{}; uint32_t macroDimBricks_{};
    std::vector<uint64_t> macroKeysHost_;
    std::vector<uint32_t> macroValsHost_;
    std::vector<uint32_t> paletteHost_;
    std::vector<world::MaterialGpu> materialTableHost_;
    VkQueryPool timestampPool_{};
    double timestampPeriodNs_{}; // GPU timestamp period in nanoseconds
    double gpuTimingsMs_[4]{};   // generate, traverse, shade, composite
    uint32_t lastTimingFrame_{}; // last frame we logged timings

    // Debug buffer (GPU→CPU) for per-frame diagnostics
    VkBuffer dbgBuf_{}; VkDeviceMemory dbgMem_{}; // 16*4 bytes sufficient
    uint32_t lastDbgFrame_{}; int lastMcX_{}; int lastMcY_{}; int lastMcZ_{}; int lastPresent_{};
    uint32_t currFrameIdx_{};
    glm::vec3 renderOrigin_{0.0f};
    VkBuffer overlayBuf_{}; VkDeviceMemory overlayMem_{}; VkDeviceSize overlayCapacity_{};
    uint32_t overlayCharsX_ = 0;
    uint32_t overlayCharsY_ = 0;
    uint32_t overlayPixelWidth_ = 0;
    uint32_t overlayPixelHeight_ = 0;
    bool overlayActive_ = false;

    std::unique_ptr<world::BrickStore> brickStore_;
    world::CpuWorld aggregateWorld_;
    bool descriptorsReady_ = false;
    bool worldDescriptorsDirty_ = false;

    bool uploadWorld(platform::VulkanContext& vk, const world::CpuWorld& cpu);
    bool rebuildGpuWorld(platform::VulkanContext& vk);
    static uint64_t packRegionKey(const glm::ivec3& coord);
    core::UploadContext uploadCtx_;
    uint32_t materialCount_ = 0;
    world::WorldGen::NoiseParams noiseParams_{};
    uint32_t worldSeed_ = 1337u;
    std::unordered_map<uint64_t, world::CpuWorld> regionWorlds_;
};

}
