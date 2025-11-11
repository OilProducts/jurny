#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/vec3.hpp>

#include "core/Upload.h"
#include "platform/VulkanContext.h"
#include "platform/Swapchain.h"
#include "world/BrickStore.h"
#include "render/GpuBuffers.h"
#include "render/Denoiser.h"
#include "render/Tonemap.h"
#include "render/Overlays.h"

// Raytracer — compute skeleton (shade sky → composite). M1 scaffolding.
namespace core { class AssetRegistry; }

namespace render {

struct GlobalsUBOData {
    float currView[16];
    float currProj[16];
    float currViewInv[16];
    float currProjInv[16];
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
    float noiseCaveThreshold, noiseMinHeight, noiseMaxHeight, noiseDetailWarp;
    float noiseSlopeSampleDist, noiseBaseHeightOffset, noisePad2, noisePad3;
    uint32_t noiseSeed, noiseContinentOctaves, noiseDetailOctaves, noiseCaveOctaves;
};
static_assert(sizeof(GlobalsUBOData) % 16 == 0, "GlobalsUBOData must align to 16 bytes");

class Raytracer {
public:
    Raytracer();
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
    size_t residentRegionCount() const { return regionResidents_.size(); }
    std::array<double, 4> gpuTimingsMs() const;
    void updateOverlayHUD(platform::VulkanContext& vk, const std::vector<std::string>& lines);
    void collectGpuTimings(platform::VulkanContext& vk, uint32_t frameIdx);

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
        VkDescriptorBufferInfo descriptor{};
        uint32_t binding = UINT32_MAX;
        bool tracked = false;
        bool dirty = false;
    };
    bool ensureBuffer(platform::VulkanContext& vk, BufferResource& buf, VkDeviceSize requiredBytes, VkBufferUsageFlags usage, bool* reallocated = nullptr);
    void destroyBuffer(platform::VulkanContext& vk, BufferResource& buf);
    void registerWorldBuffer(BufferResource& buf, uint32_t binding);
    void markWorldDescriptorsDirty();
    void refreshWorldDescriptors(platform::VulkanContext& vk);
    void markWorldBufferDirty(BufferResource& buf);
    bool appendRegion(platform::VulkanContext& vk, world::CpuWorld&& cpu, const glm::ivec3& regionCoord);
    bool removeRegionInternal(platform::VulkanContext& vk, const glm::ivec3& regionCoord);
    void rebuildHashesAndMacro(platform::VulkanContext& vk);
    void uploadAllWorldBuffers(platform::VulkanContext& vk);
    void uploadHeadersRange(platform::VulkanContext& vk, uint32_t first, uint32_t count);
    void uploadOccupancyRange(platform::VulkanContext& vk, uint32_t first, uint32_t count);
    void uploadMaterialRange(platform::VulkanContext& vk, uint32_t first, uint32_t count);
    void uploadPaletteRange(platform::VulkanContext& vk, uint32_t first, uint32_t count);
    void uploadFieldRange(platform::VulkanContext& vk, uint32_t first, uint32_t count);
    void updateBrickHeader(uint32_t index);
    void fixupBrickHeader(uint32_t index);
    bool createProfilingResources(platform::VulkanContext& vk);
   void destroyProfilingResources(platform::VulkanContext& vk);
    void updateFrameDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap, uint32_t swapIndex);

private:
    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout      pipeLayout_{};
    VkPipeline            pipeGenerate_{};
    VkPipeline            pipeShade_{};
    VkPipeline            pipeTemporal_{};
    VkPipeline            pipeTraverse_{};
    VkPipeline            pipeComposite_{};
    VkPipeline            pipeAtrous_{};
    VkPipeline            pipeOverlay_{};
    VkDescriptorPool      descPool_{};
    std::vector<VkDescriptorSet> sets_;

    VkBuffer ubo_{}; VkDeviceMemory uboMem_{};
    VkExtent2D extent_{};

    const core::AssetRegistry* assets_ = nullptr;

    // World buffers
    BufferResource bhBuf_{};
    BufferResource occBuf_{};
    BufferResource hkBuf_{};
    BufferResource hvBuf_{};
    BufferResource mkBuf_{};
    BufferResource mvBuf_{};
    BufferResource paletteBuf_{};
    BufferResource fieldBuf_{};
    BufferResource matIdxBuf_{};
    BufferResource materialTableBuf_{};
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
    uint32_t macroCapacity_{};
    std::vector<uint64_t> macroKeysHost_;
    std::vector<uint32_t> macroValsHost_;
    std::vector<world::MaterialGpu> materialTableHost_;
    VkQueryPool timestampPool_{};
    double timestampPeriodNs_{}; // GPU timestamp period in nanoseconds
    double gpuTimingsMs_[4]{};   // generate, traverse, shade, composite
    uint32_t lastTimingFrame_{}; // last frame we logged timings
    bool useSync2_ = false;

    // Debug buffer (GPU→CPU) for per-frame diagnostics
    VkBuffer dbgBuf_{}; VkDeviceMemory dbgMem_{}; // 16*4 bytes sufficient
    uint32_t lastDbgFrame_{}; int lastMcX_{}; int lastMcY_{}; int lastMcZ_{}; int lastPresent_{};
    uint32_t currFrameIdx_{};
    glm::vec3 renderOrigin_{0.0f};
    GlobalsUBOData globalsCpu_{};
    GpuBuffers gpuBuffers_;
    Denoiser denoiser_;
    bool denoiseEnabled_ = false;
    Tonemap tonemap_;
    Overlays overlays_;

    struct BrickRecord {
        uint64_t key = 0;
        glm::ivec3 coord{0};
        uint16_t paletteCount = 0;
        uint16_t flags = 0;
        bool hasField = false;
    };

    struct RegionResident {
        glm::ivec3 coord{0};
        std::vector<uint64_t> brickKeys;
    };

    std::unique_ptr<world::BrickStore> brickStore_;
    bool descriptorsReady_ = false;
    bool worldDescriptorsDirty_ = false;

    static uint64_t packRegionKey(const glm::ivec3& coord);
    core::UploadContext uploadCtx_;
    uint32_t materialCount_ = 0;
    world::WorldGen::NoiseParams noiseParams_{};
    uint32_t worldSeed_ = 1337u;
    std::unordered_map<uint64_t, RegionResident> regionResidents_;
    std::unordered_map<uint64_t, uint32_t> brickLookup_;
    std::vector<BrickRecord> brickRecords_;
    std::vector<world::BrickHeader> headersHost_;
    std::vector<uint64_t> occWordsHost_;
    std::vector<uint32_t> matWordsHost_;
    std::vector<uint32_t> paletteHost_;
    std::vector<float>    fieldHost_;
    std::vector<uint64_t> hashKeysHost_;
    std::vector<uint32_t> hashValsHost_;

    std::vector<BufferResource*> worldBuffers_;
    uint32_t macroDimBricks_ = 8;
    static constexpr uint32_t kOccWordsPerBrick = 8;
    static constexpr uint32_t kMaterialWordsPerBrick = 128;
    static constexpr uint32_t kPaletteEntriesPerBrick = 16;
    static constexpr uint32_t kFieldValuesPerBrick =
        (VOXEL_BRICK_SIZE + 1 + 2 * world::kFieldApron) *
        (VOXEL_BRICK_SIZE + 1 + 2 * world::kFieldApron) *
        (VOXEL_BRICK_SIZE + 1 + 2 * world::kFieldApron);
};

}
