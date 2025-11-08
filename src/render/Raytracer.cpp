#include "Raytracer.h"
#include "world/BrickStore.h"
#include "math/Spherical.h"
#include "core/Assets.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <utility>
#include <unordered_map>
#include <memory>

namespace render {

namespace {
constexpr uint32_t kOverlayMaxCols = 64u;
constexpr uint32_t kOverlayMaxRows = 12u;
constexpr uint32_t kOverlayFontWidth = 6u;
constexpr uint32_t kOverlayFontHeight = 8u;
constexpr uint32_t kOverlayPadX = 1u;
constexpr uint32_t kOverlayPadY = 1u;
constexpr VkDeviceSize kOverlayBufferBytes = (4u + kOverlayMaxCols * kOverlayMaxRows) * sizeof(uint32_t);
constexpr bool kCacheFieldSamples = true;
}

struct RayDispatchConstants {
    uint32_t queueSrc;
    uint32_t queueDst;
    uint32_t bounceIndex;
    uint32_t bounceCount;
};

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i) {
        if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
    }
    return UINT32_MAX;
}

static bool allocateBuffer(VkDevice device,
                           VkPhysicalDevice phys,
                           VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags flags,
                           VkBuffer& outBuf,
                           VkDeviceMemory& outMem) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &outBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device, outBuf, &mr);
    uint32_t typeIndex = findMemoryType(phys, mr.memoryTypeBits, flags);
    if (typeIndex == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(device, &mai, nullptr, &outMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, outBuf, outMem, 0);
    return true;
}

static constexpr uint32_t kTimestampCount = 8;
static constexpr uint32_t kTimestampPairs = 4;

static uint64_t packMacroKey(int x, int y, int z) {
    const uint64_t B = 1ull << 20;
    return ((uint64_t)(x + static_cast<int>(B)) << 42) |
           ((uint64_t)(y + static_cast<int>(B)) << 21) |
            (uint64_t)(z + static_cast<int>(B));
}

static bool macroTilePresentCpu(const std::vector<uint64_t>& keys,
                                const std::vector<uint32_t>& vals,
                                uint32_t capacity,
                                int x, int y, int z) {
    if (capacity == 0u || keys.empty()) return false;
    uint64_t key = packMacroKey(x, y, z);
    uint32_t mask = capacity - 1u;
    uint64_t kx = key ^ (key >> 33);
    const uint64_t A = 0xff51afd7ed558ccdULL;
    uint32_t h = static_cast<uint32_t>((kx * A) >> 32) & mask;
    for (uint32_t probe = 0; probe < capacity; ++probe) {
        uint32_t idx = (h + probe) & mask;
        uint64_t stored = keys[idx];
        if (stored == key) return vals[idx] != 0u;
        if (stored == 0ull) break;
    }
    return false;
}

bool Raytracer::ensureBuffer(platform::VulkanContext& vk,
                             BufferResource& buf,
                             VkDeviceSize requiredBytes,
                             VkBufferUsageFlags usage,
                             bool* reallocated) {
    VkDeviceSize allocSize = std::max<VkDeviceSize>(requiredBytes, static_cast<VkDeviceSize>(16));
    VkBufferUsageFlags requiredUsage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (buf.buffer != VK_NULL_HANDLE && buf.capacity >= allocSize && buf.usage == requiredUsage) {
        buf.capacity = std::max(buf.capacity, allocSize);
        if (reallocated) *reallocated = false;
        return true;
    }
    destroyBuffer(vk, buf);
    VkBuffer newBuf = VK_NULL_HANDLE;
    VkDeviceMemory newMem = VK_NULL_HANDLE;
    if (!allocateBuffer(vk.device(), vk.physicalDevice(), allocSize, requiredUsage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        newBuf, newMem)) {
        return false;
    }
    buf.buffer = newBuf;
    buf.memory = newMem;
    buf.capacity = allocSize;
    buf.size = 0;
    buf.usage = requiredUsage;
    if (reallocated) *reallocated = true;
    return true;
}

void Raytracer::destroyBuffer(platform::VulkanContext& vk, BufferResource& buf) {
    if (buf.buffer) {
        vkDestroyBuffer(vk.device(), buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }
    if (buf.memory) {
        vkFreeMemory(vk.device(), buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
    buf.capacity = 0;
    buf.size = 0;
    buf.usage = 0;
}

void Raytracer::markWorldDescriptorsDirty() {
    worldDescriptorsDirty_ = true;
}

void Raytracer::refreshWorldDescriptors(platform::VulkanContext& vk) {
    if (!descriptorsReady_ || !worldDescriptorsDirty_ || sets_.empty()) {
        return;
    }
    worldDescriptorsDirty_ = false;
    auto bufferRange = [](const BufferResource& buf) -> VkDeviceSize {
        return buf.size > 0 ? buf.size : VK_WHOLE_SIZE;
    };

    const size_t bindingCountPerSet = 10; // bindings 3..8 plus material/palette/field buffers
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(sets_.size() * bindingCountPerSet);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(sets_.size() * bindingCountPerSet);

    auto appendWrite = [&](VkDescriptorSet set, uint32_t binding, const BufferResource& buf) {
        VkDescriptorBufferInfo info{};
        info.buffer = buf.buffer;
        info.offset = 0;
        info.range = bufferRange(buf);
        infos.push_back(info);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &infos.back();
        writes.push_back(write);
    };

    for (VkDescriptorSet set : sets_) {
        appendWrite(set, 3, bhBuf_);
        appendWrite(set, 4, occBuf_);
        appendWrite(set, 5, hkBuf_);
        appendWrite(set, 6, hvBuf_);
        appendWrite(set, 7, mkBuf_);
        appendWrite(set, 8, mvBuf_);
        appendWrite(set, 23, matIdxBuf_);
        appendWrite(set, 24, materialTableBuf_);
        appendWrite(set, 25, paletteBuf_);
        appendWrite(set, 26, fieldBuf_);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(vk.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

bool Raytracer::createImages(platform::VulkanContext& vk, platform::Swapchain& swap) {
    extent_ = swap.extent();
    if (!gpuBuffers_.init(vk, extent_)) return false;
    if (denoiseEnabled_) {
        if (!denoiser_.init(vk, extent_)) return false;
    }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk.commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb{};
    vkAllocateCommandBuffers(vk.device(), &cbai, &cb);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    bool sync2 = vk.deviceInfo().hasSynchronization2;
    if (sync2) {
        std::vector<VkImageMemoryBarrier2> barriers;
        auto addBarrier = [&barriers](VkImage image) {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barrier.image = image;
            barriers.push_back(barrier);
        };

        addBarrier(gpuBuffers_.colorImage());
        addBarrier(gpuBuffers_.motionImage());
        addBarrier(gpuBuffers_.albedoImage());
        addBarrier(gpuBuffers_.normalImage());
        addBarrier(gpuBuffers_.momentsImage());
        if (denoiseEnabled_) {
            addBarrier(denoiser_.historyReadImage());
            addBarrier(denoiser_.historyWriteImage());
            addBarrier(denoiser_.historyMomentsReadImage());
            addBarrier(denoiser_.historyMomentsWriteImage());
        }

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cb, &dep);
    } else {
        std::vector<VkImageMemoryBarrier> barriers;
        auto addBarrierLegacy = [&barriers](VkImage image) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barrier.image = image;
            barriers.push_back(barrier);
        };

        addBarrierLegacy(gpuBuffers_.colorImage());
        addBarrierLegacy(gpuBuffers_.motionImage());
        addBarrierLegacy(gpuBuffers_.albedoImage());
        addBarrierLegacy(gpuBuffers_.normalImage());
        addBarrierLegacy(gpuBuffers_.momentsImage());
        if (denoiseEnabled_) {
            addBarrierLegacy(denoiser_.historyReadImage());
            addBarrierLegacy(denoiser_.historyWriteImage());
            addBarrierLegacy(denoiser_.historyMomentsReadImage());
            addBarrierLegacy(denoiser_.historyMomentsWriteImage());
        }

        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              0, nullptr,
                              0, nullptr,
                              static_cast<uint32_t>(barriers.size()), barriers.data());
    }
    vkEndCommandBuffer(cb);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    vkQueueSubmit(vk.graphicsQueue(), 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk.graphicsQueue());
    vkFreeCommandBuffers(vk.device(), vk.commandPool(), 1, &cb);
    return true;
}

void Raytracer::destroyImages(platform::VulkanContext& vk) {
    if (denoiseEnabled_) {
        denoiser_.shutdown(vk);
    }
    gpuBuffers_.shutdown(vk);
}

bool Raytracer::createPipelines(platform::VulkanContext& vk) {
    // Descriptor set layout: 0=UBO, 1=currColor, 2=out, 3=BrickHeaders, 4=Occ, 5=HashKeys, 6=HashVals,
    // 7=MacroKeys, 8=MacroVals, 9=Debug, 10=RayQueue, 11=HitQueue, 12=MissQueue, 13=SecondaryQueue,
    // 14=TraversalStats, 15=MotionImage, 16=HistoryColorRead, 17=HistoryColorWrite,
    // 18=Albedo, 19=NormalDepth, 20=MomentsCurrent, 21=HistoryMomentsRead, 22=HistoryMomentsWrite
    VkDescriptorSetLayoutBinding bUbo{ }; bUbo.binding=0; bUbo.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; bUbo.descriptorCount=1; bUbo.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bAcc{ }; bAcc.binding=1; bAcc.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bAcc.descriptorCount=1; bAcc.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOut{ }; bOut.binding=2; bOut.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bOut.descriptorCount=1; bOut.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bBH{ };  bBH.binding=3;  bBH.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bBH.descriptorCount=1; bBH.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOcc{ }; bOcc.binding=4; bOcc.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bOcc.descriptorCount=1; bOcc.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHK{ };  bHK.binding=5;  bHK.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bHK.descriptorCount=1; bHK.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHV{ };  bHV.binding=6;  bHV.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bHV.descriptorCount=1; bHV.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMK{ };  bMK.binding=7;  bMK.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMK.descriptorCount=1; bMK.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMV{ };  bMV.binding=8;  bMV.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMV.descriptorCount=1; bMV.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bDBG{ }; bDBG.binding=9;  bDBG.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bDBG.descriptorCount=1; bDBG.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bRayQ{ }; bRayQ.binding=10; bRayQ.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bRayQ.descriptorCount=1; bRayQ.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHitQ{ }; bHitQ.binding=11; bHitQ.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bHitQ.descriptorCount=1; bHitQ.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMissQ{}; bMissQ.binding=12; bMissQ.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMissQ.descriptorCount=1; bMissQ.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bSecQ{ }; bSecQ.binding=13; bSecQ.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bSecQ.descriptorCount=1; bSecQ.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bStats{ }; bStats.binding=14; bStats.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bStats.descriptorCount=1; bStats.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMotion{ }; bMotion.binding=15; bMotion.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bMotion.descriptorCount=1; bMotion.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHistRead{ }; bHistRead.binding=16; bHistRead.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bHistRead.descriptorCount=1; bHistRead.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHistWrite{ }; bHistWrite.binding=17; bHistWrite.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bHistWrite.descriptorCount=1; bHistWrite.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bAlbedo{ }; bAlbedo.binding=18; bAlbedo.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bAlbedo.descriptorCount=1; bAlbedo.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bNormal{ }; bNormal.binding=19; bNormal.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bNormal.descriptorCount=1; bNormal.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMoments{ }; bMoments.binding=20; bMoments.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bMoments.descriptorCount=1; bMoments.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHistMomentsRead{ }; bHistMomentsRead.binding=21; bHistMomentsRead.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bHistMomentsRead.descriptorCount=1; bHistMomentsRead.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHistMomentsWrite{ }; bHistMomentsWrite.binding=22; bHistMomentsWrite.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bHistMomentsWrite.descriptorCount=1; bHistMomentsWrite.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMatIdx{ }; bMatIdx.binding=23; bMatIdx.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMatIdx.descriptorCount=1; bMatIdx.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMatTable{ }; bMatTable.binding=24; bMatTable.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMatTable.descriptorCount=1; bMatTable.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bPalette{ }; bPalette.binding=25; bPalette.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bPalette.descriptorCount=1; bPalette.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bField{ }; bField.binding=26; bField.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bField.descriptorCount=1; bField.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOverlayBuf{}; bOverlayBuf.binding=27; bOverlayBuf.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bOverlayBuf.descriptorCount=1; bOverlayBuf.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOverlayImage{}; bOverlayImage.binding=28; bOverlayImage.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bOverlayImage.descriptorCount=1; bOverlayImage.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[29] = { bUbo, bAcc, bOut, bBH, bOcc, bHK, bHV, bMK, bMV, bDBG, bRayQ, bHitQ, bMissQ, bSecQ, bStats, bMotion, bHistRead, bHistWrite, bAlbedo, bNormal, bMoments, bHistMomentsRead, bHistMomentsWrite, bMatIdx, bMatTable, bPalette, bField, bOverlayBuf, bOverlayImage };
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(sizeof(bindings) / sizeof(bindings[0]));
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk.device(), &dslci, nullptr, &setLayout_) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(RayDispatchConstants);
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(vk.device(), &plci, nullptr, &pipeLayout_) != VK_SUCCESS) return false;

    auto readFile = [](const char* path) -> std::vector<uint32_t> {
        struct FileCloser {
            void operator()(FILE* f) const noexcept { if (f) std::fclose(f); }
        };
        std::vector<uint32_t> words;
        FILE* raw = std::fopen(path, "rb");
        if (!raw) {
            const int err = errno;
            spdlog::error("Failed to open shader {}: {}", path, std::strerror(err));
            return {};
        }
        std::unique_ptr<FILE, FileCloser> file(raw);
        if (std::fseek(file.get(), 0, SEEK_END) != 0) {
            const int err = errno;
            spdlog::error("Failed to seek to end of shader {}: {}", path, std::strerror(err));
            return {};
        }
        long end = std::ftell(file.get());
        if (end < 0) {
            const int err = errno;
            spdlog::error("Failed to ftell shader {}: {}", path, std::strerror(err));
            return {};
        }
        if (std::fseek(file.get(), 0, SEEK_SET) != 0) {
            const int err = errno;
            spdlog::error("Failed to rewind shader {}: {}", path, std::strerror(err));
            return {};
        }
        const size_t size = static_cast<size_t>(end);
        if (size == 0) {
            spdlog::error("Shader {} is empty", path);
            return {};
        }
        if ((size % sizeof(uint32_t)) != 0u) {
            spdlog::error("Shader {} has byte size {} which is not aligned to 4 bytes", path, size);
            return {};
        }
        words.resize(size / sizeof(uint32_t));
        const size_t read = std::fread(words.data(), 1, size, file.get());
        if (read != size) {
            if (std::ferror(file.get())) {
                const int err = errno;
                spdlog::error("Failed while reading shader {}: {}", path, std::strerror(err));
            } else {
                spdlog::error("Unexpected EOF while reading shader {} (expected {} bytes, got {})", path, size, read);
            }
            return {};
        }
        return words;
    };
    auto loadShader = [&](const char* rel)->VkShaderModule{
        const char* envAssets = std::getenv("VOXEL_ASSETS_DIR");
        const char* assetsDir = envAssets ? envAssets :
#ifdef VOXEL_ASSETS_DIR
            VOXEL_ASSETS_DIR;
#else
            "assets";
#endif
        std::string base = std::string(assetsDir) + "/shaders/";
        std::vector<uint32_t> spv = readFile((base + rel).c_str());
        if (spv.empty()) return VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode = spv.data();
        VkShaderModule sm{}; vkCreateShaderModule(vk.device(), &smci, nullptr, &sm); return sm;
    };
    VkShaderModule smGen   = loadShader("generate_rays.comp.spv");
    VkShaderModule smShade = loadShader("shade.comp.spv");
    VkShaderModule smTemporal = loadShader("denoise_atrous.comp.spv");
    VkShaderModule smTrav  = loadShader("traverse_bricks.comp.spv");
    VkShaderModule smComp  = loadShader("composite.comp.spv");
    VkShaderModule smOverlay = loadShader("overlay.comp.spv");
    if (!smGen || !smShade || !smTemporal || !smTrav || !smComp || !smOverlay) return false;
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    VkPipelineShaderStageCreateInfo ss{};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.pName = "main";
    cpci.layout = pipeLayout_;
    ss.module = smGen;   cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeGenerate_) != VK_SUCCESS) return false;
    ss.module = smShade; cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeShade_) != VK_SUCCESS) return false;
    ss.module = smTemporal; cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeTemporal_) != VK_SUCCESS) return false;
    ss.module = smTrav;  cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeTraverse_) != VK_SUCCESS) return false;
    ss.module = smComp;  cpci.stage = ss;
    if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeComposite_) != VK_SUCCESS) return false;
    ss.module = smOverlay; cpci.stage = ss;
    if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeOverlay_) != VK_SUCCESS) return false;
    vkDestroyShaderModule(vk.device(), smGen, nullptr);
    vkDestroyShaderModule(vk.device(), smShade, nullptr);
    vkDestroyShaderModule(vk.device(), smTemporal, nullptr);
    vkDestroyShaderModule(vk.device(), smTrav, nullptr);
    vkDestroyShaderModule(vk.device(), smComp, nullptr);
    vkDestroyShaderModule(vk.device(), smOverlay, nullptr);
    return true;
}

void Raytracer::destroyPipelines(platform::VulkanContext& vk) {
    if (pipeComposite_) { vkDestroyPipeline(vk.device(), pipeComposite_, nullptr); pipeComposite_ = VK_NULL_HANDLE; }
    if (pipeTemporal_) { vkDestroyPipeline(vk.device(), pipeTemporal_, nullptr); pipeTemporal_ = VK_NULL_HANDLE; }
    if (pipeShade_) { vkDestroyPipeline(vk.device(), pipeShade_, nullptr); pipeShade_ = VK_NULL_HANDLE; }
    if (pipeTraverse_) { vkDestroyPipeline(vk.device(), pipeTraverse_, nullptr); pipeTraverse_ = VK_NULL_HANDLE; }
    if (pipeGenerate_) { vkDestroyPipeline(vk.device(), pipeGenerate_, nullptr); pipeGenerate_ = VK_NULL_HANDLE; }
    if (pipeOverlay_) { vkDestroyPipeline(vk.device(), pipeOverlay_, nullptr); pipeOverlay_ = VK_NULL_HANDLE; }
    if (pipeLayout_) { vkDestroyPipelineLayout(vk.device(), pipeLayout_, nullptr); pipeLayout_ = VK_NULL_HANDLE; }
    if (setLayout_) { vkDestroyDescriptorSetLayout(vk.device(), setLayout_, nullptr); setLayout_ = VK_NULL_HANDLE; }
}

bool Raytracer::createDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap) {
    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   12u * swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  18u * swap.imageCount() }
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = swap.imageCount();
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &descPool_) != VK_SUCCESS) return false;
    sets_.resize(swap.imageCount()); std::vector<VkDescriptorSetLayout> layouts(swap.imageCount(), setLayout_);
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = swap.imageCount();
    dsai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(vk.device(), &dsai, sets_.data()) != VK_SUCCESS) return false;

    // Create UBO
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = sizeof(GlobalsUBOData);
    bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (vkCreateBuffer(vk.device(), &bi, nullptr, &ubo_) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), ubo_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &uboMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), ubo_, uboMem_, 0);

    // Create Debug buffer (host visible)
    VkBufferCreateInfo bdbg{};
    bdbg.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bdbg.size = 128 * sizeof(uint32_t); // expanded for richer diagnostics
    bdbg.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(vk.device(), &bdbg, nullptr, &dbgBuf_) != VK_SUCCESS) return false;
    VkMemoryRequirements mrd{}; vkGetBufferMemoryRequirements(vk.device(), dbgBuf_, &mrd);
    uint32_t typeIndexDbg = findMemoryType(vk.physicalDevice(), mrd.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo maid{};
    maid.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    maid.allocationSize = mrd.size;
    maid.memoryTypeIndex = typeIndexDbg;
    if (vkAllocateMemory(vk.device(), &maid, nullptr, &dbgMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), dbgBuf_, dbgMem_, 0);

    if (!createProfilingResources(vk)) return false;
    if (!overlays_.init(vk, kOverlayBufferBytes)) return false;
    if (VkDeviceMemory overlayMem = overlays_.memory()) {
        void* data = nullptr;
        if (vkMapMemory(vk.device(), overlayMem, 0, overlays_.capacity(), 0, &data) == VK_SUCCESS && data) {
            std::memset(data, 0, static_cast<size_t>(overlays_.capacity()));
            vkUnmapMemory(vk.device(), overlayMem);
        }
    }
    if (matIdxBuf_.buffer == VK_NULL_HANDLE) {
    if (!ensureBuffer(vk, matIdxBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr)) return false;
    }
    if (paletteBuf_.buffer == VK_NULL_HANDLE) {
    if (!ensureBuffer(vk, paletteBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr)) return false;
    }
    if (fieldBuf_.buffer == VK_NULL_HANDLE) {
    if (!ensureBuffer(vk, fieldBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr)) return false;
    }
    if (materialTableBuf_.buffer == VK_NULL_HANDLE) {
    if (!ensureBuffer(vk, materialTableBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr)) return false;
    }

    // Write descriptors per swapchain image
    for (uint32_t i=0;i<swap.imageCount();++i) {
        VkDescriptorBufferInfo db{}; db.buffer = ubo_; db.offset=0; db.range = sizeof(GlobalsUBOData);
        VkDescriptorImageInfo diCurr{}; diCurr.imageView = gpuBuffers_.colorView(); diCurr.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diOut{}; diOut.imageView = swap.imageViews()[i]; diOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diMotion{}; diMotion.imageView = gpuBuffers_.motionView(); diMotion.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistRead{}; diHistRead.imageView = denoiser_.historyReadView(); diHistRead.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistWrite{}; diHistWrite.imageView = denoiser_.historyWriteView(); diHistWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diAlbedo{}; diAlbedo.imageView = gpuBuffers_.albedoView(); diAlbedo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diNormal{}; diNormal.imageView = gpuBuffers_.normalView(); diNormal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diMoments{}; diMoments.imageView = gpuBuffers_.momentsView(); diMoments.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistMomentsRead{}; diHistMomentsRead.imageView = denoiser_.historyMomentsReadView(); diHistMomentsRead.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistMomentsWrite{}; diHistMomentsWrite.imageView = denoiser_.historyMomentsWriteView(); diHistMomentsWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo dbDBG{ dbgBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbRay{ gpuBuffers_.queueBuffer(GpuBuffers::Queue::Ray), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHit{ gpuBuffers_.queueBuffer(GpuBuffers::Queue::Hit), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMiss{ gpuBuffers_.queueBuffer(GpuBuffers::Queue::Miss), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbSec{ gpuBuffers_.queueBuffer(GpuBuffers::Queue::Secondary), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbStats{ gpuBuffers_.statsBuffer(), 0, sizeof(TraversalStatsHost) };
        VkDescriptorBufferInfo dbMatIdx{ matIdxBuf_.buffer, 0, matIdxBuf_.size > 0 ? matIdxBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMatTable{ materialTableBuf_.buffer, 0, materialTableBuf_.size > 0 ? materialTableBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbPalette{ paletteBuf_.buffer, 0, paletteBuf_.size > 0 ? paletteBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbField{ fieldBuf_.buffer, 0, fieldBuf_.size > 0 ? fieldBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbOverlay{ overlays_.buffer(), 0, overlays_.capacity() ? overlays_.capacity() : VK_WHOLE_SIZE };
        VkDescriptorImageInfo diOverlay{}; diOverlay.imageView = swap.imageViews()[i]; diOverlay.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[23]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = sets_[i]; writes[0].dstBinding=0; writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].descriptorCount=1; writes[0].pBufferInfo=&db;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = sets_[i]; writes[1].dstBinding=1; writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].descriptorCount=1; writes[1].pImageInfo=&diCurr;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = sets_[i]; writes[2].dstBinding=2; writes[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[2].descriptorCount=1; writes[2].pImageInfo=&diOut;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[3].dstSet = sets_[i]; writes[3].dstBinding=15; writes[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[3].descriptorCount=1; writes[3].pImageInfo=&diMotion;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[4].dstSet = sets_[i]; writes[4].dstBinding=16; writes[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[4].descriptorCount=1; writes[4].pImageInfo=&diHistRead;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[5].dstSet = sets_[i]; writes[5].dstBinding=17; writes[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[5].descriptorCount=1; writes[5].pImageInfo=&diHistWrite;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[6].dstSet = sets_[i]; writes[6].dstBinding=9; writes[6].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[6].descriptorCount=1; writes[6].pBufferInfo=&dbDBG;
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[7].dstSet=sets_[i]; writes[7].dstBinding=10; writes[7].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[7].descriptorCount=1; writes[7].pBufferInfo=&dbRay;
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[8].dstSet=sets_[i]; writes[8].dstBinding=11; writes[8].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[8].descriptorCount=1; writes[8].pBufferInfo=&dbHit;
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[9].dstSet=sets_[i]; writes[9].dstBinding=12; writes[9].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[9].descriptorCount=1; writes[9].pBufferInfo=&dbMiss;
        writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[10].dstSet=sets_[i]; writes[10].dstBinding=13; writes[10].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[10].descriptorCount=1; writes[10].pBufferInfo=&dbSec;
        writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[11].dstSet=sets_[i]; writes[11].dstBinding=14; writes[11].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[11].descriptorCount=1; writes[11].pBufferInfo=&dbStats;
        writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[12].dstSet=sets_[i]; writes[12].dstBinding=18; writes[12].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[12].descriptorCount=1; writes[12].pImageInfo=&diAlbedo;
        writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[13].dstSet=sets_[i]; writes[13].dstBinding=19; writes[13].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[13].descriptorCount=1; writes[13].pImageInfo=&diNormal;
        writes[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[14].dstSet=sets_[i]; writes[14].dstBinding=20; writes[14].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[14].descriptorCount=1; writes[14].pImageInfo=&diMoments;
        writes[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[15].dstSet=sets_[i]; writes[15].dstBinding=21; writes[15].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[15].descriptorCount=1; writes[15].pImageInfo=&diHistMomentsRead;
        writes[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[16].dstSet=sets_[i]; writes[16].dstBinding=22; writes[16].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[16].descriptorCount=1; writes[16].pImageInfo=&diHistMomentsWrite;
        writes[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[17].dstSet=sets_[i]; writes[17].dstBinding=23; writes[17].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[17].descriptorCount=1; writes[17].pBufferInfo=&dbMatIdx;
        writes[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[18].dstSet=sets_[i]; writes[18].dstBinding=24; writes[18].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[18].descriptorCount=1; writes[18].pBufferInfo=&dbMatTable;
        writes[19].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[19].dstSet=sets_[i]; writes[19].dstBinding=25; writes[19].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[19].descriptorCount=1; writes[19].pBufferInfo=&dbPalette;
        writes[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[20].dstSet=sets_[i]; writes[20].dstBinding=26; writes[20].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[20].descriptorCount=1; writes[20].pBufferInfo=&dbField;
        writes[21].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[21].dstSet=sets_[i]; writes[21].dstBinding=27; writes[21].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[21].descriptorCount=1; writes[21].pBufferInfo=&dbOverlay;
        writes[22].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[22].dstSet=sets_[i]; writes[22].dstBinding=28; writes[22].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[22].descriptorCount=1; writes[22].pImageInfo=&diOverlay;
        uint32_t writeCount = static_cast<uint32_t>(sizeof(writes) / sizeof(writes[0]));
        vkUpdateDescriptorSets(vk.device(), writeCount, writes, 0, nullptr);
    }
    descriptorsReady_ = true;
    refreshWorldDescriptors(vk);
    return true;
}

void Raytracer::destroyDescriptors(platform::VulkanContext& vk) {
    destroyProfilingResources(vk);
    if (ubo_) { vkDestroyBuffer(vk.device(), ubo_, nullptr); ubo_ = VK_NULL_HANDLE; }
    if (uboMem_) { vkFreeMemory(vk.device(), uboMem_, nullptr); uboMem_ = VK_NULL_HANDLE; }
    if (dbgBuf_) { vkDestroyBuffer(vk.device(), dbgBuf_, nullptr); dbgBuf_ = VK_NULL_HANDLE; }
    if (dbgMem_) { vkFreeMemory(vk.device(), dbgMem_, nullptr); dbgMem_ = VK_NULL_HANDLE; }
    overlays_.shutdown(vk);
    if (descPool_) { vkDestroyDescriptorPool(vk.device(), descPool_, nullptr); descPool_ = VK_NULL_HANDLE; }
    sets_.clear();
    descriptorsReady_ = false;
    if (denoiseEnabled_) {
        denoiser_.reset();
    }
}

bool Raytracer::createWorld(platform::VulkanContext& vk) {
    brickStore_ = std::make_unique<world::BrickStore>();
    auto& store = *brickStore_;
    math::PlanetParams P{ 100.0, 120.0, 100.0, 160.0 }; // base radius, trench depth, sea level, max height
    math::NoiseTuning tuning{};
    tuning.continentsPerCircumference = 3.2f;
    tuning.continentAmplitude         = 110.0f;
    tuning.continentOctaves           = 5;
    tuning.detailWavelength           = 55.0f;
    tuning.detailAmplitude            = 16.0f;
    tuning.detailOctaves              = 3;
    tuning.detailWarpMultiplier       = 1.8f;
    tuning.baseHeightOffset           = 14.0f;
    tuning.warpWavelength             = 240.0f;
    tuning.warpAmplitude              = 24.0f;
    tuning.slopeSampleDistance        = 100.0f;
    tuning.caveWavelength             = 36.0f;
    tuning.caveAmplitude              = 5.0f;
    tuning.caveThreshold              = 0.35f;
    tuning.moistureWavelength         = 80.0f;
    tuning.moistureOctaves            = 4;
    noiseParams_ = math::BuildNoiseParams(tuning, P);
    worldSeed_ = 1337u;
    store.configure(P, /*voxelSize*/0.5f, /*brickDim*/VOXEL_BRICK_SIZE, noiseParams_, worldSeed_, assets_);

    brickRecords_.clear();
    headersHost_.clear();
    occWordsHost_.clear();
    matWordsHost_.clear();
    paletteHost_.clear();
    fieldHost_.clear();
    brickLookup_.clear();
    regionResidents_.clear();
    macroKeysHost_.clear();
    macroValsHost_.clear();
    macroCapacity_ = 0;
    hashCapacity_ = 0;
    brickCount_ = 0;
    materialCount_ = 0;
    macroDimBricks_ = 8;
    spdlog::info("Initial world bricks: 0 (0 MiB occupancy)");
    markWorldDescriptorsDirty();
    (void)vk;
    return true;
}

uint64_t Raytracer::packRegionKey(const glm::ivec3& coord) {
    const uint64_t B = 1ull << 20;
    return (static_cast<uint64_t>(coord.x + static_cast<int>(B)) << 42) |
           (static_cast<uint64_t>(coord.y + static_cast<int>(B)) << 21) |
            static_cast<uint64_t>(coord.z + static_cast<int>(B));
}

namespace {
uint64_t packBrickKey(int bx, int by, int bz) {
    const uint64_t B = 1ull << 20;
    return (static_cast<uint64_t>(bx + static_cast<int>(B)) << 42) |
           (static_cast<uint64_t>(by + static_cast<int>(B)) << 21) |
            static_cast<uint64_t>(bz + static_cast<int>(B));
}

int divFloor(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if (((a ^ b) < 0) && (r != 0)) --q;
    return q;
}
}

bool Raytracer::addRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord, world::CpuWorld&& cpu) {
    if (cpu.headers.empty()) {
        return true;
    }
    if (!appendRegion(vk, std::move(cpu), regionCoord)) {
        return false;
    }
    if (denoiseEnabled_) {
        denoiser_.reset();
    }
    return true;
}

bool Raytracer::removeRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord) {
    if (!removeRegionInternal(vk, regionCoord)) {
        return false;
    }
    if (denoiseEnabled_) {
        denoiser_.reset();
    }
    return true;
}

bool Raytracer::appendRegion(platform::VulkanContext& vk, world::CpuWorld&& cpu, const glm::ivec3& regionCoord) {
    const auto regionStart = std::chrono::steady_clock::now();
    const uint32_t oldCount = static_cast<uint32_t>(headersHost_.size());
    const uint32_t requested = static_cast<uint32_t>(cpu.headers.size());
    if (requested == 0u) {
        regionResidents_.emplace(packRegionKey(regionCoord), RegionResident{regionCoord, {}});
        return true;
    }

    const uint32_t voxelCount = VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE;
    const uint32_t words4Bit = (voxelCount + 7u) / 8u;
    const uint32_t words8Bit = (voxelCount + 3u) / 4u;

    if (cpu.macroDimBricks > 0u) {
        macroDimBricks_ = cpu.macroDimBricks;
    }

    headersHost_.resize(static_cast<size_t>(oldCount + requested));
    brickRecords_.resize(static_cast<size_t>(oldCount + requested));
    occWordsHost_.resize(static_cast<size_t>(oldCount + requested) * kOccWordsPerBrick);
    matWordsHost_.resize(static_cast<size_t>(oldCount + requested) * kMaterialWordsPerBrick);
    paletteHost_.resize(static_cast<size_t>(oldCount + requested) * kPaletteEntriesPerBrick);
    if (kCacheFieldSamples) {
        fieldHost_.resize(static_cast<size_t>(oldCount + requested) * kFieldValuesPerBrick);
    } else {
        fieldHost_.clear();
    }

    RegionResident resident;
    resident.coord = regionCoord;
    resident.brickKeys.reserve(requested);

    for (uint32_t i = 0; i < requested; ++i) {
        uint32_t dstIndex = oldCount + i;
        const world::BrickHeader& srcHeader = cpu.headers[i];
        uint64_t key = packBrickKey(srcHeader.bx, srcHeader.by, srcHeader.bz);
        if (brickLookup_.find(key) != brickLookup_.end()) {
            spdlog::warn("Skipping duplicate brick at ({}, {}, {})", srcHeader.bx, srcHeader.by, srcHeader.bz);
            continue;
        }

        BrickRecord record;
        record.key = key;
        record.coord = glm::ivec3(srcHeader.bx, srcHeader.by, srcHeader.bz);
        record.paletteCount = srcHeader.paletteCount;
        record.flags = srcHeader.flags;
        record.hasField = kCacheFieldSamples && (srcHeader.tsdfOffset != world::kInvalidOffset);
        brickLookup_[key] = dstIndex;
        resident.brickKeys.push_back(key);

        world::BrickHeader header{};
        header.bx = srcHeader.bx;
        header.by = srcHeader.by;
        header.bz = srcHeader.bz;
        header.flags = srcHeader.flags;
        header.paletteCount = srcHeader.paletteCount;
        header.occOffset = dstIndex * kOccWordsPerBrick * sizeof(uint64_t);
        header.matIdxOffset = dstIndex * kMaterialWordsPerBrick * sizeof(uint32_t);
        header.paletteOffset = (header.paletteCount > 0) ? dstIndex * kPaletteEntriesPerBrick * sizeof(uint32_t) : world::kInvalidOffset;
        header.tsdfOffset = world::kInvalidOffset;
        headersHost_[dstIndex] = header;

        const uint64_t* occSrc = cpu.occWords.data() + (srcHeader.occOffset / sizeof(uint64_t));
        uint64_t* occDst = occWordsHost_.data() + static_cast<size_t>(dstIndex) * kOccWordsPerBrick;
        std::copy_n(occSrc, kOccWordsPerBrick, occDst);

        const uint32_t* matSrc = cpu.materialIndices.data() + (srcHeader.matIdxOffset / sizeof(uint32_t));
        uint32_t* matDst = matWordsHost_.data() + static_cast<size_t>(dstIndex) * kMaterialWordsPerBrick;
        std::fill(matDst, matDst + kMaterialWordsPerBrick, 0u);
        const bool uses4bit = (srcHeader.flags & world::kBrickUses4Bit) != 0;
        const uint32_t wordCount = uses4bit ? words4Bit : words8Bit;
        std::copy_n(matSrc, wordCount, matDst);

        uint32_t* paletteDst = paletteHost_.data() + static_cast<size_t>(dstIndex) * kPaletteEntriesPerBrick;
        std::fill(paletteDst, paletteDst + kPaletteEntriesPerBrick, 0u);
        if (srcHeader.paletteCount > 0 && srcHeader.paletteOffset != world::kInvalidOffset) {
            const uint32_t* paletteSrc = cpu.palettes.data() + (srcHeader.paletteOffset / sizeof(uint32_t));
            std::copy_n(paletteSrc, srcHeader.paletteCount, paletteDst);
        }

        if (kCacheFieldSamples) {
            float* fieldDst = fieldHost_.data() + static_cast<size_t>(dstIndex) * kFieldValuesPerBrick;
            std::fill(fieldDst, fieldDst + kFieldValuesPerBrick, 0.0f);
            if (record.hasField && srcHeader.tsdfOffset != world::kInvalidOffset) {
                size_t srcOffset = static_cast<size_t>(srcHeader.tsdfOffset / sizeof(float));
                if (srcOffset + kFieldValuesPerBrick <= cpu.fieldSamples.size()) {
                    const float* fieldSrc = cpu.fieldSamples.data() + srcOffset;
                    std::copy_n(fieldSrc, kFieldValuesPerBrick, fieldDst);
                } else {
                    spdlog::warn("TSDF field truncated for brick ({}, {}, {}) — expected {} floats, have {}", srcHeader.bx, srcHeader.by, srcHeader.bz, kFieldValuesPerBrick, cpu.fieldSamples.size() - srcOffset);
                    record.hasField = false;
                    std::fill(fieldDst, fieldDst + kFieldValuesPerBrick, 0.0f);
                }
            }
        } else {
            record.hasField = false;
        }

        brickRecords_[dstIndex] = record;

        fixupBrickHeader(dstIndex);
    }

    const uint32_t appended = static_cast<uint32_t>(resident.brickKeys.size());
    const uint32_t newCount = oldCount + appended;
    headersHost_.resize(newCount);
    brickRecords_.resize(newCount);
    occWordsHost_.resize(static_cast<size_t>(newCount) * kOccWordsPerBrick);
    matWordsHost_.resize(static_cast<size_t>(newCount) * kMaterialWordsPerBrick);
    paletteHost_.resize(static_cast<size_t>(newCount) * kPaletteEntriesPerBrick);
    if (kCacheFieldSamples) {
        fieldHost_.resize(static_cast<size_t>(newCount) * kFieldValuesPerBrick);
    } else {
        fieldHost_.clear();
    }

    if (materialTableHost_.empty() && !cpu.materialTable.empty()) {
        materialTableHost_ = cpu.materialTable;
        materialCount_ = static_cast<uint32_t>(materialTableHost_.size());
        bool reallocated = false;
        VkDeviceSize totalBytes = materialTableHost_.size() * sizeof(world::MaterialGpu);
        if (!ensureBuffer(vk, materialTableBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
            return false;
        }
        materialTableBuf_.size = totalBytes;
        if (totalBytes > 0) {
            uploadCtx_.uploadBufferRegion(materialTableHost_.data(), totalBytes, materialTableBuf_.buffer, 0);
        }
    }

    regionResidents_.emplace(packRegionKey(regionCoord), std::move(resident));
    brickCount_ = newCount;
    const auto packEnd = std::chrono::steady_clock::now();
    const double packMs = std::chrono::duration<double, std::milli>(packEnd - regionStart).count();

    const auto uploadBegin = std::chrono::steady_clock::now();
    if (appended > 0) {
        uploadHeadersRange(vk, oldCount, appended);
        uploadOccupancyRange(vk, oldCount, appended);
        uploadMaterialRange(vk, oldCount, appended);
        uploadPaletteRange(vk, oldCount, appended);
        uploadFieldRange(vk, oldCount, appended);
    }
    const auto uploadEnd = std::chrono::steady_clock::now();
    const double uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadBegin).count();

    const auto hashStart = std::chrono::steady_clock::now();
    rebuildHashesAndMacro(vk);
    const auto regionEnd = std::chrono::steady_clock::now();
    const double hashMs = std::chrono::duration<double, std::milli>(regionEnd - hashStart).count();
    const double totalMs = std::chrono::duration<double, std::milli>(regionEnd - regionStart).count();
    spdlog::info("Raytracer append region ({}, {}, {}): total {:.2f} ms [pack {:.2f} ms | upload {:.2f} ms | hash {:.2f} ms], bricks now {} (+{})",
                 regionCoord.x, regionCoord.y, regionCoord.z,
                 totalMs, packMs, uploadMs, hashMs,
                 brickCount_, appended);
    markWorldDescriptorsDirty();
    return true;
}

bool Raytracer::removeRegionInternal(platform::VulkanContext& vk, const glm::ivec3& regionCoord) {
    const uint64_t regionKey = packRegionKey(regionCoord);
    auto it = regionResidents_.find(regionKey);
    if (it == regionResidents_.end()) {
        return true;
    }

    for (uint64_t key : it->second.brickKeys) {
        auto lookup = brickLookup_.find(key);
        if (lookup == brickLookup_.end()) {
            continue;
        }
        uint32_t idx = lookup->second;
        uint32_t last = static_cast<uint32_t>(headersHost_.size() - 1);
        if (idx != last) {
            headersHost_[idx] = headersHost_[last];
            brickRecords_[idx] = brickRecords_[last];
            brickLookup_[brickRecords_[idx].key] = idx;

            std::copy_n(occWordsHost_.data() + static_cast<size_t>(last) * kOccWordsPerBrick,
                        kOccWordsPerBrick,
                        occWordsHost_.data() + static_cast<size_t>(idx) * kOccWordsPerBrick);
            std::copy_n(matWordsHost_.data() + static_cast<size_t>(last) * kMaterialWordsPerBrick,
                        kMaterialWordsPerBrick,
                        matWordsHost_.data() + static_cast<size_t>(idx) * kMaterialWordsPerBrick);
            std::copy_n(paletteHost_.data() + static_cast<size_t>(last) * kPaletteEntriesPerBrick,
                        kPaletteEntriesPerBrick,
                        paletteHost_.data() + static_cast<size_t>(idx) * kPaletteEntriesPerBrick);
            if (kCacheFieldSamples) {
                std::copy_n(fieldHost_.data() + static_cast<size_t>(last) * kFieldValuesPerBrick,
                            kFieldValuesPerBrick,
                            fieldHost_.data() + static_cast<size_t>(idx) * kFieldValuesPerBrick);
            }
            fixupBrickHeader(idx);
        }

        headersHost_.pop_back();
        brickRecords_.pop_back();
        brickLookup_.erase(lookup);
        occWordsHost_.resize(static_cast<size_t>(brickRecords_.size()) * kOccWordsPerBrick);
        matWordsHost_.resize(static_cast<size_t>(brickRecords_.size()) * kMaterialWordsPerBrick);
        paletteHost_.resize(static_cast<size_t>(brickRecords_.size()) * kPaletteEntriesPerBrick);
        if (kCacheFieldSamples) {
            fieldHost_.resize(static_cast<size_t>(brickRecords_.size()) * kFieldValuesPerBrick);
        } else {
            fieldHost_.clear();
        }
    }

    regionResidents_.erase(it);
    brickCount_ = static_cast<uint32_t>(headersHost_.size());
    uploadAllWorldBuffers(vk);
    rebuildHashesAndMacro(vk);
    markWorldDescriptorsDirty();
    return true;
}

void Raytracer::uploadHeadersRange(platform::VulkanContext& vk, uint32_t first, uint32_t count) {
    VkDeviceSize totalBytes = static_cast<VkDeviceSize>(headersHost_.size()) * sizeof(world::BrickHeader);
    bool reallocated = false;
    if (!ensureBuffer(vk, bhBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    bhBuf_.size = totalBytes;
    if (totalBytes == 0) {
        return;
    }
    if (reallocated || count == headersHost_.size()) {
        uploadCtx_.uploadBufferRegion(headersHost_.data(), totalBytes, bhBuf_.buffer, 0);
    } else if (count > 0) {
        const world::BrickHeader* src = headersHost_.data() + first;
        VkDeviceSize offset = static_cast<VkDeviceSize>(first) * sizeof(world::BrickHeader);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(world::BrickHeader);
        uploadCtx_.uploadBufferRegion(src, bytes, bhBuf_.buffer, offset);
    }
}

void Raytracer::uploadOccupancyRange(platform::VulkanContext& vk, uint32_t first, uint32_t count) {
    VkDeviceSize totalBytes = static_cast<VkDeviceSize>(occWordsHost_.size()) * sizeof(uint64_t);
    bool reallocated = false;
    if (!ensureBuffer(vk, occBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    occBuf_.size = totalBytes;
    if (totalBytes == 0) {
        return;
    }
    if (reallocated || count == brickCount_) {
        uploadCtx_.uploadBufferRegion(occWordsHost_.data(), totalBytes, occBuf_.buffer, 0);
    } else if (count > 0) {
        const uint64_t* src = occWordsHost_.data() + static_cast<size_t>(first) * kOccWordsPerBrick;
        VkDeviceSize offset = static_cast<VkDeviceSize>(first) * kOccWordsPerBrick * sizeof(uint64_t);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * kOccWordsPerBrick * sizeof(uint64_t);
        uploadCtx_.uploadBufferRegion(src, bytes, occBuf_.buffer, offset);
    }
}

void Raytracer::uploadMaterialRange(platform::VulkanContext& vk, uint32_t first, uint32_t count) {
    VkDeviceSize totalBytes = static_cast<VkDeviceSize>(matWordsHost_.size()) * sizeof(uint32_t);
    bool reallocated = false;
    if (!ensureBuffer(vk, matIdxBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    matIdxBuf_.size = totalBytes;
    if (totalBytes == 0) {
        return;
    }
    if (reallocated || count == brickCount_) {
        uploadCtx_.uploadBufferRegion(matWordsHost_.data(), totalBytes, matIdxBuf_.buffer, 0);
    } else if (count > 0) {
        const uint32_t* src = matWordsHost_.data() + static_cast<size_t>(first) * kMaterialWordsPerBrick;
        VkDeviceSize offset = static_cast<VkDeviceSize>(first) * kMaterialWordsPerBrick * sizeof(uint32_t);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * kMaterialWordsPerBrick * sizeof(uint32_t);
        uploadCtx_.uploadBufferRegion(src, bytes, matIdxBuf_.buffer, offset);
    }
}

void Raytracer::uploadPaletteRange(platform::VulkanContext& vk, uint32_t first, uint32_t count) {
    VkDeviceSize totalBytes = static_cast<VkDeviceSize>(paletteHost_.size()) * sizeof(uint32_t);
    bool reallocated = false;
    if (!ensureBuffer(vk, paletteBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    paletteBuf_.size = totalBytes;
    if (totalBytes == 0) {
        return;
    }
    if (reallocated || count == brickCount_) {
        uploadCtx_.uploadBufferRegion(paletteHost_.data(), totalBytes, paletteBuf_.buffer, 0);
    } else if (count > 0) {
        const uint32_t* src = paletteHost_.data() + static_cast<size_t>(first) * kPaletteEntriesPerBrick;
        VkDeviceSize offset = static_cast<VkDeviceSize>(first) * kPaletteEntriesPerBrick * sizeof(uint32_t);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * kPaletteEntriesPerBrick * sizeof(uint32_t);
        uploadCtx_.uploadBufferRegion(src, bytes, paletteBuf_.buffer, offset);
    }
}

void Raytracer::uploadFieldRange(platform::VulkanContext& vk, uint32_t first, uint32_t count) {
    if (!kCacheFieldSamples) {
        fieldBuf_.size = 0;
        return;
    }
    VkDeviceSize totalBytes = static_cast<VkDeviceSize>(fieldHost_.size()) * sizeof(float);
    bool reallocated = false;
    if (!ensureBuffer(vk, fieldBuf_, totalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    fieldBuf_.size = totalBytes;
    if (totalBytes == 0) {
        return;
    }
    if (reallocated || count == brickCount_) {
        uploadCtx_.uploadBufferRegion(fieldHost_.data(), totalBytes, fieldBuf_.buffer, 0);
    } else if (count > 0) {
        const float* src = fieldHost_.data() + static_cast<size_t>(first) * kFieldValuesPerBrick;
        VkDeviceSize offset = static_cast<VkDeviceSize>(first) * kFieldValuesPerBrick * sizeof(float);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * kFieldValuesPerBrick * sizeof(float);
        uploadCtx_.uploadBufferRegion(src, bytes, fieldBuf_.buffer, offset);
    }
}

void Raytracer::uploadAllWorldBuffers(platform::VulkanContext& vk) {
    if (brickCount_ == 0u) {
        bhBuf_.size = 0;
        occBuf_.size = 0;
        matIdxBuf_.size = 0;
        paletteBuf_.size = 0;
        fieldBuf_.size = 0;
        return;
    }
    uploadHeadersRange(vk, 0, brickCount_);
    uploadOccupancyRange(vk, 0, brickCount_);
    uploadMaterialRange(vk, 0, brickCount_);
    uploadPaletteRange(vk, 0, brickCount_);
    uploadFieldRange(vk, 0, brickCount_);
}

void Raytracer::rebuildHashesAndMacro(platform::VulkanContext& vk) {
    hashKeysHost_.clear();
    hashValsHost_.clear();
    macroKeysHost_.clear();
    macroValsHost_.clear();

    if (brickCount_ == 0u) {
        hashCapacity_ = 0;
        macroCapacity_ = 0;
        hkBuf_.size = 0;
        hvBuf_.size = 0;
        mkBuf_.size = 0;
        mvBuf_.size = 0;
        return;
    }

    hashCapacity_ = 1u;
    while (hashCapacity_ < brickCount_ * 2u) hashCapacity_ <<= 1u;
    if (hashCapacity_ < 8u) hashCapacity_ = 8u;

    hashKeysHost_.assign(hashCapacity_, 0ull);
    hashValsHost_.assign(hashCapacity_, 0u);
    const uint32_t mask = hashCapacity_ - 1u;
    const uint64_t A = 0xff51afd7ed558ccdULL;
    for (uint32_t i = 0; i < brickCount_; ++i) {
        uint64_t key = brickRecords_[i].key;
        uint64_t kx = key ^ (key >> 33);
        uint32_t h = static_cast<uint32_t>((kx * A) >> 32) & mask;
        for (uint32_t probe = 0; probe < hashCapacity_; ++probe) {
            uint32_t idx = (h + probe) & mask;
            if (hashKeysHost_[idx] == 0ull) {
                hashKeysHost_[idx] = key;
                hashValsHost_[idx] = i;
                break;
            }
        }
    }

    bool reallocated = false;
    VkDeviceSize hashKeyBytes = static_cast<VkDeviceSize>(hashKeysHost_.size()) * sizeof(uint64_t);
    if (!ensureBuffer(vk, hkBuf_, hashKeyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    hkBuf_.size = hashKeyBytes;
    if (hashKeyBytes > 0) {
        uploadCtx_.uploadBufferRegion(hashKeysHost_.data(), hashKeyBytes, hkBuf_.buffer, 0);
    }

    reallocated = false;
    VkDeviceSize hashValBytes = static_cast<VkDeviceSize>(hashValsHost_.size()) * sizeof(uint32_t);
    if (!ensureBuffer(vk, hvBuf_, hashValBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    hvBuf_.size = hashValBytes;
    if (hashValBytes > 0) {
        uploadCtx_.uploadBufferRegion(hashValsHost_.data(), hashValBytes, hvBuf_.buffer, 0);
    }

    std::unordered_map<uint64_t, uint32_t> macroCounts;
    macroCounts.reserve(brickCount_);
    for (const auto& record : brickRecords_) {
        int mx = divFloor(record.coord.x, static_cast<int>(macroDimBricks_));
        int my = divFloor(record.coord.y, static_cast<int>(macroDimBricks_));
        int mz = divFloor(record.coord.z, static_cast<int>(macroDimBricks_));
        ++macroCounts[packMacroKey(mx, my, mz)];
    }

    if (macroCounts.empty()) {
        macroCapacity_ = 0;
        mkBuf_.size = 0;
        mvBuf_.size = 0;
        macroKeysHost_.clear();
        macroValsHost_.clear();
        return;
    }

    macroCapacity_ = 1u;
    while (macroCapacity_ < macroCounts.size() * 2u) macroCapacity_ <<= 1u;
    if (macroCapacity_ < 8u) macroCapacity_ = 8u;
    macroKeysHost_.assign(macroCapacity_, 0ull);
    macroValsHost_.assign(macroCapacity_, 0u);
    const uint32_t macroMask = macroCapacity_ - 1u;
    for (const auto& kv : macroCounts) {
        uint64_t key = kv.first;
        uint64_t kx = key ^ (key >> 33);
        uint32_t h = static_cast<uint32_t>((kx * A) >> 32) & macroMask;
        for (uint32_t probe = 0; probe < macroCapacity_; ++probe) {
            uint32_t idx = (h + probe) & macroMask;
            if (macroKeysHost_[idx] == 0ull) {
                macroKeysHost_[idx] = key;
                macroValsHost_[idx] = kv.second;
                break;
            }
        }
    }

    reallocated = false;
    VkDeviceSize macroKeyBytes = static_cast<VkDeviceSize>(macroKeysHost_.size()) * sizeof(uint64_t);
    if (!ensureBuffer(vk, mkBuf_, macroKeyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    mkBuf_.size = macroKeyBytes;
    if (macroKeyBytes > 0) {
        uploadCtx_.uploadBufferRegion(macroKeysHost_.data(), macroKeyBytes, mkBuf_.buffer, 0);
    }

    reallocated = false;
    VkDeviceSize macroValBytes = static_cast<VkDeviceSize>(macroValsHost_.size()) * sizeof(uint32_t);
    if (!ensureBuffer(vk, mvBuf_, macroValBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reallocated)) {
        return;
    }
    mvBuf_.size = macroValBytes;
    if (macroValBytes > 0) {
        uploadCtx_.uploadBufferRegion(macroValsHost_.data(), macroValBytes, mvBuf_.buffer, 0);
    }
}

void Raytracer::updateBrickHeader(uint32_t index) {
    fixupBrickHeader(index);
}

void Raytracer::fixupBrickHeader(uint32_t index) {
    if (index >= headersHost_.size() || index >= brickRecords_.size()) {
        return;
    }
    world::BrickHeader& header = headersHost_[index];
    const BrickRecord& record = brickRecords_[index];
    header.occOffset = index * kOccWordsPerBrick * sizeof(uint64_t);
    header.matIdxOffset = index * kMaterialWordsPerBrick * sizeof(uint32_t);
    header.paletteCount = record.paletteCount;
    header.flags = record.flags;
    header.paletteOffset = (record.paletteCount > 0) ? index * kPaletteEntriesPerBrick * sizeof(uint32_t) : world::kInvalidOffset;
    header.tsdfOffset = record.hasField ? index * kFieldValuesPerBrick * sizeof(float) : world::kInvalidOffset;
}

void Raytracer::destroyWorld(platform::VulkanContext& vk) {
    destroyBuffer(vk, bhBuf_);
    destroyBuffer(vk, occBuf_);
    destroyBuffer(vk, hkBuf_);
    destroyBuffer(vk, hvBuf_);
    destroyBuffer(vk, mkBuf_);
    destroyBuffer(vk, mvBuf_);
    destroyBuffer(vk, paletteBuf_);
    destroyBuffer(vk, fieldBuf_);
    destroyBuffer(vk, matIdxBuf_);
    destroyBuffer(vk, materialTableBuf_);
    macroKeysHost_.clear();
    macroValsHost_.clear();
    hashKeysHost_.clear();
    hashValsHost_.clear();
    paletteHost_.clear();
    fieldHost_.clear();
    materialTableHost_.clear();
    brickRecords_.clear();
    headersHost_.clear();
    occWordsHost_.clear();
    matWordsHost_.clear();
    brickLookup_.clear();
    regionResidents_.clear();
    brickCount_ = 0;
    hashCapacity_ = 0;
    macroCapacity_ = 0;
    macroDimBricks_ = 0;
    materialCount_ = 0;
    markWorldDescriptorsDirty();
}

namespace {
}


bool Raytracer::createProfilingResources(platform::VulkanContext& vk) {
    destroyProfilingResources(vk);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vk.physicalDevice(), &props);
    timestampPeriodNs_ = props.limits.timestampPeriod;
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kTimestampCount;
    if (vkCreateQueryPool(vk.device(), &qpci, nullptr, &timestampPool_) != VK_SUCCESS) {
        timestampPool_ = VK_NULL_HANDLE;
        return false;
    }
    std::fill(std::begin(gpuTimingsMs_), std::end(gpuTimingsMs_), 0.0);
    lastTimingFrame_ = 0;
    return true;
}

void Raytracer::destroyProfilingResources(platform::VulkanContext& vk) {
    if (timestampPool_) {
        vkDestroyQueryPool(vk.device(), timestampPool_, nullptr);
        timestampPool_ = VK_NULL_HANDLE;
    }
}

void Raytracer::updateFrameDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap, uint32_t swapIndex) {
    if (!descriptorsReady_) return;

    VkDescriptorImageInfo diCurr{ VK_NULL_HANDLE, gpuBuffers_.colorView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diMotion{ VK_NULL_HANDLE, gpuBuffers_.motionView(), VK_IMAGE_LAYOUT_GENERAL };
    VkImageView historyReadView  = denoiseEnabled_ ? denoiser_.historyReadView() : gpuBuffers_.colorView();
    VkImageView historyWriteView = denoiseEnabled_ ? denoiser_.historyWriteView() : gpuBuffers_.colorView();
    VkImageView historyMomentsReadView  = denoiseEnabled_ ? denoiser_.historyMomentsReadView() : gpuBuffers_.momentsView();
    VkImageView historyMomentsWriteView = denoiseEnabled_ ? denoiser_.historyMomentsWriteView() : gpuBuffers_.momentsView();
    VkDescriptorImageInfo diHistRead{ VK_NULL_HANDLE, historyReadView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistWrite{ VK_NULL_HANDLE, historyWriteView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diMoments{ VK_NULL_HANDLE, gpuBuffers_.momentsView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistMomentsRead{ VK_NULL_HANDLE, historyMomentsReadView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistMomentsWrite{ VK_NULL_HANDLE, historyMomentsWriteView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo diOverlay{ VK_NULL_HANDLE, swap.imageViews()[swapIndex], VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 8> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sets_[swapIndex];
    writes[0].dstBinding = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &diCurr;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sets_[swapIndex];
    writes[1].dstBinding = 15;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &diMotion;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = sets_[swapIndex];
    writes[2].dstBinding = 16;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &diHistRead;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = sets_[swapIndex];
    writes[3].dstBinding = 17;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &diHistWrite;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = sets_[swapIndex];
    writes[4].dstBinding = 20;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].descriptorCount = 1;
    writes[4].pImageInfo = &diMoments;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = sets_[swapIndex];
    writes[5].dstBinding = 21;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].descriptorCount = 1;
    writes[5].pImageInfo = &diHistMomentsRead;

    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = sets_[swapIndex];
    writes[6].dstBinding = 22;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[6].descriptorCount = 1;
    writes[6].pImageInfo = &diHistMomentsWrite;

    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = sets_[swapIndex];
    writes[7].dstBinding = 28;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[7].descriptorCount = 1;
    writes[7].pImageInfo = &diOverlay;

    vkUpdateDescriptorSets(vk.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool Raytracer::init(platform::VulkanContext& vk, platform::Swapchain& swap) {
    useSync2_ = vk.deviceInfo().hasSynchronization2;
    if (!createImages(vk, swap)) return false;
    if (!createPipelines(vk)) return false;
    if (!uploadCtx_.init(vk)) return false;
    if (!createWorld(vk)) return false;
    if (!createDescriptors(vk, swap)) return false;
    return true;
}

void Raytracer::resize(platform::VulkanContext& vk, platform::Swapchain& swap) {
    vkDeviceWaitIdle(vk.device());
    destroyDescriptors(vk);
    destroyImages(vk);
    createImages(vk, swap);
    createDescriptors(vk, swap);
}

void Raytracer::updateGlobals(platform::VulkanContext& vk, const GlobalsUBOData& data) {
    GlobalsUBOData d = data;
    d.worldHashCapacity = hashCapacity_;
    d.worldBrickCount   = brickCount_;
    d.macroHashCapacity = macroCapacity_;
    d.macroDimBricks    = macroDimBricks_;
    d.macroSize         = float(macroDimBricks_) * (d.brickSize);
    d.historyValid      = denoiseEnabled_ ? denoiser_.historyValidFlag() : 0u;
    if (brickStore_) {
        const auto& wg = brickStore_->worldGen();
        const auto& np = wg.params();
        d.noiseContinentFreq = np.continentFrequency;
        d.noiseContinentAmp  = np.continentAmplitude;
        d.noiseDetailFreq    = np.detailFrequency;
        d.noiseDetailAmp     = np.detailAmplitude;
        d.noiseWarpFreq      = np.warpFrequency;
        d.noiseWarpAmp       = np.warpAmplitude;
        d.noiseCaveFreq      = np.caveFrequency;
        d.noiseCaveAmp       = np.caveAmplitude;
        d.noiseCaveThreshold = np.caveThreshold;
        d.noiseMinHeight     = -static_cast<float>(wg.planet().T);
        d.noiseMaxHeight     =  static_cast<float>(wg.planet().Hmax);
        d.noiseDetailWarp    = np.detailWarpMultiplier;
        d.noiseSlopeSampleDist = np.slopeSampleDistance;
        d.noiseBaseHeightOffset = np.baseHeightOffset;
        d.noisePad2          = 0.0f;
        d.noisePad3          = 0.0f;
        d.noiseSeed          = wg.seed();
        d.noiseContinentOctaves = static_cast<uint32_t>(np.continentOctaves);
        d.noiseDetailOctaves    = static_cast<uint32_t>(np.detailOctaves);
        d.noiseCaveOctaves      = static_cast<uint32_t>(math::kNoiseCaveOctaves);
    } else {
        d.noiseContinentFreq = d.noiseContinentAmp = 0.0f;
        d.noiseDetailFreq = d.noiseDetailAmp = 0.0f;
        d.noiseWarpFreq = d.noiseWarpAmp = 0.0f;
        d.noiseCaveFreq = d.noiseCaveAmp = 0.0f;
        d.noiseCaveThreshold = 0.0f;
        d.noiseMinHeight = d.noiseMaxHeight = 0.0f;
        d.noiseDetailWarp = 0.0f;
        d.noiseSlopeSampleDist = 0.0f;
        d.noiseBaseHeightOffset = 0.0f;
        d.noisePad2 = d.noisePad3 = 0.0f;
        d.noiseSeed = 0u;
        d.noiseContinentOctaves = d.noiseDetailOctaves = d.noiseCaveOctaves = 0u;
    }
    renderOrigin_ = glm::vec3(d.renderOrigin[0], d.renderOrigin[1], d.renderOrigin[2]);
    tonemap_.setExposure(d.exposure);
    void* mapped=nullptr; vkMapMemory(vk.device(), uboMem_, 0, sizeof(GlobalsUBOData), 0, &mapped);
    std::memcpy(mapped, &d, sizeof(GlobalsUBOData));
    vkUnmapMemory(vk.device(), uboMem_);
    currFrameIdx_ = d.frameIdx;
    globalsCpu_ = d;
}

void Raytracer::record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex) {
    constexpr bool kDisableRayDispatch = false;
    constexpr bool kEnableGenerateStage = true;
    constexpr bool kEnableTraverseStage = true;
    constexpr bool kEnableShadeStage = true;
    if (kDisableRayDispatch) {
        return;
    }
    refreshWorldDescriptors(vk);
    updateFrameDescriptors(vk, swap, swapIndex);
    gpuBuffers_.writeQueueHeaders(cb);
    gpuBuffers_.zeroStats(cb);
    const uint32_t bounceCount = std::max(1u, globalsCpu_.maxBounces == 0u ? 1u : globalsCpu_.maxBounces);
    auto pushDispatchConstants = [&](uint32_t queueSrc, uint32_t queueDst, uint32_t bounceIndex) {
        RayDispatchConstants constants{queueSrc, queueDst, bounceIndex, bounceCount};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayDispatchConstants), &constants);
    };
    auto queueFromIndex = [](uint32_t idx) {
        return (idx == 0u) ? GpuBuffers::Queue::Ray : GpuBuffers::Queue::Secondary;
    };
    auto issueBarrier = [&](VkPipelineStageFlags2 srcStage2,
                            VkAccessFlags2 srcAccess2,
                            VkPipelineStageFlags2 dstStage2,
                            VkAccessFlags2 dstAccess2,
                            VkPipelineStageFlags srcStage,
                            VkAccessFlags srcAccess,
                            VkPipelineStageFlags dstStage,
                            VkAccessFlags dstAccess) {
        if (useSync2_) {
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.srcStageMask = srcStage2;
            barrier.srcAccessMask = srcAccess2;
            barrier.dstStageMask = dstStage2;
            barrier.dstAccessMask = dstAccess2;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cb, &dep);
        } else {
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = srcAccess;
            barrier.dstAccessMask = dstAccess;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
    };

    auto writeTimestamp = [&](uint32_t index) {
        if (!timestampPool_) return;
        if (useSync2_) {
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, index);
        } else {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool_, index);
        }
    };

    issueBarrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (timestampPool_) {
        vkCmdResetQueryPool(cb, timestampPool_, 0, kTimestampCount);
        writeTimestamp(0);
    }

    // Generate primary rays into queue
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    uint32_t gx = (extent_.width + 7u) / 8u;
    uint32_t gy = (extent_.height + 7u) / 8u;
    pushDispatchConstants(0u, 1u, 0u);
    if (kEnableGenerateStage) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeGenerate_);
        vkCmdDispatch(cb, gx, gy, 1);
        writeTimestamp(1);
    }
    if (kEnableGenerateStage && (kEnableTraverseStage || kEnableShadeStage)) {
        issueBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        writeTimestamp(2);
    }

    for (uint32_t bounce = 0; bounce < bounceCount; ++bounce) {
        const uint32_t srcIdx = (bounce & 1u) == 0u ? 0u : 1u;
        const uint32_t dstIdx = (bounce & 1u) == 0u ? 1u : 0u;
        pushDispatchConstants(srcIdx, dstIdx, bounce);
        gpuBuffers_.resetQueueHeader(cb, queueFromIndex(dstIdx));
        gpuBuffers_.resetQueueHeader(cb, GpuBuffers::Queue::Hit);
        gpuBuffers_.resetQueueHeader(cb, GpuBuffers::Queue::Miss);

        if (kEnableTraverseStage) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTraverse_);
            vkCmdDispatch(cb, gx, gy, 1);
        }

        if (kEnableTraverseStage && kEnableShadeStage) {
            issueBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            if (bounce == 0u) {
                writeTimestamp(3);
            }
        }

        if (kEnableShadeStage) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeShade_);
            uint32_t shadeGroups = gx * gy;
            if (bounce == 0u) {
                writeTimestamp(4);
            }
            vkCmdDispatch(cb, shadeGroups, 1, 1);
            if (bounce == 0u && timestampPool_) {
                vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 5);
            }
        }
    }

    // Make current color and motion available to downstream passes
    std::vector<VkImageMemoryBarrier2> preTemporal;
    preTemporal.reserve(8);
    auto makeBarrier = [](VkImage image) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        return barrier;
    };
    preTemporal.push_back(makeBarrier(gpuBuffers_.colorImage()));
    preTemporal.push_back(makeBarrier(gpuBuffers_.motionImage()));
    preTemporal.push_back(makeBarrier(gpuBuffers_.albedoImage()));
    preTemporal.push_back(makeBarrier(gpuBuffers_.normalImage()));
    preTemporal.push_back(makeBarrier(gpuBuffers_.momentsImage()));
    if (denoiseEnabled_) {
        preTemporal.push_back(makeBarrier(denoiser_.historyReadImage()));
        preTemporal.push_back(makeBarrier(denoiser_.historyWriteImage()));
        preTemporal.push_back(makeBarrier(denoiser_.historyMomentsReadImage()));
        preTemporal.push_back(makeBarrier(denoiser_.historyMomentsWriteImage()));
    }
    VkDependencyInfo preTemporalDep{};
    preTemporalDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preTemporalDep.imageMemoryBarrierCount = static_cast<uint32_t>(preTemporal.size());
    preTemporalDep.pImageMemoryBarriers = preTemporal.data();
    vkCmdPipelineBarrier2(cb, &preTemporalDep);

    if (denoiseEnabled_) {
        // Temporal accumulation into history write target
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTemporal_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
        uint32_t tx = (extent_.width + 7u) / 8u;
        uint32_t ty = (extent_.height + 7u) / 8u;
        vkCmdDispatch(cb, tx, ty, 1);

        // Barrier history image for composite pass
        std::array<VkImageMemoryBarrier2, 2> histBarriers{};
        histBarriers[0] = makeBarrier(denoiser_.historyWriteImage());
        histBarriers[1] = makeBarrier(denoiser_.historyMomentsWriteImage());
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(histBarriers.size());
        dep.pImageMemoryBarriers = histBarriers.data();
        vkCmdPipelineBarrier2(cb, &dep);
    }

    // Composite: read temporally accumulated history, write swap image already in GENERAL from App
    tonemap_.record(cb,
                    pipeComposite_,
                    pipeLayout_,
                    sets_[swapIndex],
                    gx,
                    gy,
                    timestampPool_,
                    6,
                    7);

    if (denoiseEnabled_) {
        denoiser_.advance();
    }

    // Debug readback moved to readDebug(), called after submit/present to ensure GPU wrote the data.
}

void Raytracer::recordOverlay(VkCommandBuffer cb, uint32_t swapIndex) {
    if (!overlays_.active() || overlays_.pixelWidth() == 0 || overlays_.pixelHeight() == 0 || pipeOverlay_ == VK_NULL_HANDLE) {
        return;
    }
    if (swapIndex >= sets_.size()) {
        return;
    }
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cb, &dep);
    uint32_t groupsX = (overlays_.pixelWidth() + 7u) / 8u;
    uint32_t groupsY = (overlays_.pixelHeight() + 7u) / 8u;
    if (groupsX == 0u || groupsY == 0u) {
        return;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeOverlay_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    vkCmdDispatch(cb, groupsX, groupsY, 1);
}

std::array<double, 4> Raytracer::gpuTimingsMs() const {
    return { gpuTimingsMs_[0], gpuTimingsMs_[1], gpuTimingsMs_[2], gpuTimingsMs_[3] };
}

void Raytracer::updateOverlayHUD(platform::VulkanContext& vk, const std::vector<std::string>& lines) {
    std::vector<std::string> sanitized;
    sanitized.reserve(std::min<uint32_t>(static_cast<uint32_t>(lines.size()), kOverlayMaxRows));

    for (uint32_t i = 0; i < std::min<uint32_t>(static_cast<uint32_t>(lines.size()), kOverlayMaxRows); ++i) {
        const std::string& line = lines[i];
        std::string clean;
        clean.reserve(kOverlayMaxCols);
        for (char ch : line) {
            if (clean.size() >= kOverlayMaxCols) break;
            unsigned char uc = static_cast<unsigned char>(ch);
            if (uc < 32u || uc > 126u) {
                uc = ' ';
            }
            uc = static_cast<unsigned char>(std::toupper(static_cast<int>(uc)));
            clean.push_back(static_cast<char>(uc));
        }
        while (!clean.empty() && clean.back() == ' ') {
            clean.pop_back();
        }
        sanitized.emplace_back(std::move(clean));
    }

    overlays_.update(vk,
                     sanitized,
                     kOverlayMaxCols,
                     kOverlayMaxRows,
                     kOverlayFontWidth,
                     kOverlayFontHeight,
                     kOverlayPadX,
                     kOverlayPadY);
}

void Raytracer::readDebug(platform::VulkanContext& vk, uint32_t frameIdx) {
    spdlog::info("readDebug begin frame {}", frameIdx);
    // Poll infrequently and debounce identical frames to avoid log spam.
    // Temporarily read back every frame for profiling
    void* p=nullptr; if (vkMapMemory(vk.device(), dbgMem_, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) {
        spdlog::info("readDebug map failed frame {}", frameIdx);
        return;
    }
    uint32_t* u = reinterpret_cast<uint32_t*>(p);
    uint32_t frame = u[0]; uint32_t flags = u[1]; uint32_t mcap=u[2]; uint32_t mdim=u[3];
    int bcx0 = int(u[4]), bcy0=int(u[5]), bcz0=int(u[6]); uint32_t present0=u[7];
    int mcx0 = int(u[8]), mcy0=int(u[9]), mcz0=int(u[10]); uint32_t macroStepCount=u[11];
    int bcxH = int(u[12]), bcyH=int(u[13]), bczH=int(u[14]); uint32_t presentHit=u[15];
    // Extended diagnostics
    uint32_t zeroDirMask = u[16];
    uint32_t hugeStepCount = u[17];
    uint32_t nanCount = u[18];
    uint32_t breakMask = u[19];
    uint32_t zeroSentinelUseCount = u[20];
    uint32_t clampCount = u[21];
    float lastTNext = 0.f, lastTSearch = 0.f;
    std::memcpy(&lastTNext, &u[22], sizeof(float));
    std::memcpy(&lastTSearch, &u[23], sizeof(float));

    bool changed = (frame != lastDbgFrame_) || (mcx0!=lastMcX_) || (mcy0!=lastMcY_) || (mcz0!=lastMcZ_) || (int(present0)!=lastPresent_);
    if (changed) {
        spdlog::debug("DBG frame={} flags={} macroCap={} macroDim={} start-bc=({}, {}, {}) start-mc=({}, {}, {}) present={} msteps={} hit-bc=({}, {}, {}) hit-present={}",
                     frame, flags, mcap, mdim,
                     bcx0,bcy0,bcz0, mcx0,mcy0,mcz0, present0, macroStepCount,
                     bcxH,bcyH,bczH, presentHit);
        spdlog::debug("DBG macroDiag: zeroMask={} hugeCnt={} nanCnt={} breakMask={} zeroSentUses={} clampCnt={} lastTNext={:.6f} lastTSearch={:.6f}",
                     zeroDirMask, hugeStepCount, nanCount, breakMask, zeroSentinelUseCount, clampCount, lastTNext, lastTSearch);
        // Decode extended diagnostics (band start and sea-level expectations)
        int mcStartX = int(u[24]), mcStartY = int(u[25]), mcStartZ = int(u[26]); uint32_t presentStart = u[27];
        int mcSeaX   = int(u[28]), mcSeaY   = int(u[29]), mcSeaZ   = int(u[30]); uint32_t presentSea = u[31];
        int bcBandX  = int(u[32]), bcBandY  = int(u[33]), bcBandZ  = int(u[34]);
        int bcSeaX   = int(u[35]), bcSeaY   = int(u[36]), bcSeaZ   = int(u[37]);
        spdlog::debug("DBG bandStart: mc=({}, {}, {}) present={} bc=({}, {}, {}) | sea: mc=({}, {}, {}) present={} bc=({}, {}, {})",
                     mcStartX, mcStartY, mcStartZ, presentStart,
                     bcBandX, bcBandY, bcBandZ,
                     mcSeaX, mcSeaY, mcSeaZ, presentSea,
                     bcSeaX, bcSeaY, bcSeaZ);
        if (brickStore_) {
            float brickSize = brickStore_->brickSize();
            glm::vec3 startWorld = glm::vec3(float(bcx0), float(bcy0), float(bcz0)) * brickSize;
            glm::vec3 hitWorld   = glm::vec3(float(bcxH), float(bcyH), float(bczH)) * brickSize;
            glm::vec3 startLocal = startWorld - renderOrigin_;
            glm::vec3 hitLocal   = hitWorld - renderOrigin_;
            spdlog::debug("DBG brick centers: startWorld=({:.2f},{:.2f},{:.2f}) startLocal=({:.2f},{:.2f},{:.2f}) | hitWorld=({:.2f},{:.2f},{:.2f}) hitLocal=({:.2f},{:.2f},{:.2f}) renderOrigin=({:.2f},{:.2f},{:.2f})",
                          startWorld.x, startWorld.y, startWorld.z,
                          startLocal.x, startLocal.y, startLocal.z,
                          hitWorld.x, hitWorld.y, hitWorld.z,
                          hitLocal.x, hitLocal.y, hitLocal.z,
                          renderOrigin_.x, renderOrigin_.y, renderOrigin_.z);
        }
        // Times overview to verify windowing
        float tEnterF=0, tExitF=0, s0F=0, s1F=0, sNearF=0, tBandMinF=0, tBandMaxF=0, tMacroMinF=0, tMacroMaxF=0;
        std::memcpy(&tEnterF,   &u[38], sizeof(float));
        std::memcpy(&tExitF,    &u[39], sizeof(float));
        std::memcpy(&s0F,       &u[40], sizeof(float));
        std::memcpy(&s1F,       &u[41], sizeof(float));
        std::memcpy(&sNearF,    &u[42], sizeof(float));
        std::memcpy(&tBandMinF, &u[43], sizeof(float));
        std::memcpy(&tBandMaxF, &u[44], sizeof(float));
        std::memcpy(&tMacroMinF,&u[45], sizeof(float));
        std::memcpy(&tMacroMaxF,&u[46], sizeof(float));
        spdlog::debug("DBG times: tEnter={:.3f} tExit={:.3f} s0={:.3f} s1={:.3f} sNear={:.3f} | band=[{:.3f}..{:.3f}] macro=[{:.3f}..{:.3f}]",
                     tEnterF, tExitF, s0F, s1F, sNearF, tBandMinF, tBandMaxF, tMacroMinF, tMacroMaxF);
        lastDbgFrame_ = frame; lastMcX_=mcx0; lastMcY_=mcy0; lastMcZ_=mcz0; lastPresent_=int(present0);
    }
    auto decodeInt = [](uint32_t v) {
        int32_t out;
        std::memcpy(&out, &v, sizeof(out));
        return out;
    };
    uint32_t macroDiagSentinel = u[60];
    if (macroDiagSentinel != 0u) {
        int startX = decodeInt(u[61]);
        int startY = decodeInt(u[62]);
        int startZ = decodeInt(u[63]);
        uint32_t startPresent = u[64];
        uint32_t stepCount = u[65];
        int endX = decodeInt(u[66]);
        int endY = decodeInt(u[67]);
        int endZ = decodeInt(u[68]);
        uint32_t endPresent = u[69];
        uint32_t reason = u[70];
        bool cpuStart = macroTilePresentCpu(macroKeysHost_, macroValsHost_, macroCapacity_, startX, startY, startZ);
        bool cpuEnd = macroTilePresentCpu(macroKeysHost_, macroValsHost_, macroCapacity_, endX, endY, endZ);
        spdlog::debug("Macro diag: start=({}, {}, {}) present={} cpu={} steps={} end=({}, {}, {}) present={} cpu={} reason={}",
                     startX, startY, startZ, startPresent, cpuStart ? 1 : 0,
                     stepCount,
                     endX, endY, endZ, endPresent, cpuEnd ? 1 : 0,
                     reason);
    }
    vkUnmapMemory(vk.device(), dbgMem_);

    if (timestampPool_) {
        uint64_t timestamps[kTimestampCount];
        VkResult qr = vkGetQueryPoolResults(vk.device(), timestampPool_, 0, kTimestampCount,
                                            sizeof(timestamps), timestamps, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr == VK_SUCCESS) {
            double scaleMs = timestampPeriodNs_ * 1e-6;
            auto diffMs = [&](uint32_t a, uint32_t b) {
                return (timestamps[b] > timestamps[a]) ? double(timestamps[b] - timestamps[a]) * scaleMs : 0.0;
            };
            gpuTimingsMs_[0] = diffMs(0, 1);
            gpuTimingsMs_[1] = diffMs(2, 3);
            gpuTimingsMs_[2] = diffMs(4, 5);
            gpuTimingsMs_[3] = diffMs(6, 7);
            lastTimingFrame_ = frameIdx;
        }
    }

    if (VkDeviceMemory statsMem = gpuBuffers_.statsMemory()) {
        TraversalStatsHost stats{};
        void* mapped = nullptr;
        VkResult mapRes = vkMapMemory(vk.device(), statsMem, 0, sizeof(TraversalStatsHost), 0, &mapped);
        if (mapRes == VK_SUCCESS && mapped) {
            std::memcpy(&stats, mapped, sizeof(TraversalStatsHost));
            vkUnmapMemory(vk.device(), statsMem);
            statsHost_ = stats;
            spdlog::debug("Traverse stats: macroVisited={} macroSkipped={} brickSteps={} microSteps={} hits={}",
                          stats.macroVisited, stats.macroSkipped, stats.brickSteps, stats.microSteps, stats.hitsTotal);
        } else {
            spdlog::warn("Failed to map traversal stats buffer (res={})", int(mapRes));
        }
    }
    spdlog::info("readDebug end frame {}", frameIdx);
}

void Raytracer::collectGpuTimings(platform::VulkanContext& vk, uint32_t frameIdx) {
    if (!timestampPool_) return;
    uint64_t timestamps[kTimestampCount]{};
    VkResult qr = vkGetQueryPoolResults(vk.device(), timestampPool_, 0, kTimestampCount,
                                        sizeof(timestamps), timestamps, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT);
    if (qr == VK_NOT_READY) {
        return; // queries not complete yet; skip this frame
    }
    if (qr != VK_SUCCESS) {
        spdlog::warn("vkGetQueryPoolResults failed ({})", int(qr));
        return;
    }
    double scaleMs = timestampPeriodNs_ * 1e-6;
    auto diffMs = [&](uint32_t a, uint32_t b) {
        return (timestamps[b] > timestamps[a]) ? double(timestamps[b] - timestamps[a]) * scaleMs : 0.0;
    };
    gpuTimingsMs_[0] = diffMs(0, 1);
    gpuTimingsMs_[1] = diffMs(2, 3);
    gpuTimingsMs_[2] = diffMs(4, 5);
    gpuTimingsMs_[3] = diffMs(6, 7);
    lastTimingFrame_ = frameIdx;
}

void Raytracer::shutdown(platform::VulkanContext& vk) {
    destroyDescriptors(vk);
    destroyPipelines(vk);
    destroyImages(vk);
    destroyWorld(vk);
    uploadCtx_.shutdown();
    brickStore_.reset();
}

}
