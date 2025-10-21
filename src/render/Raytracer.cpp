#include "Raytracer.h"
#include "world/BrickStore.h"
#include "math/Spherical.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cctype>

namespace render {

namespace {
constexpr uint32_t kOverlayMaxCols = 64u;
constexpr uint32_t kOverlayMaxRows = 12u;
constexpr uint32_t kOverlayFontWidth = 6u;
constexpr uint32_t kOverlayFontHeight = 8u;
constexpr uint32_t kOverlayPadX = 1u;
constexpr uint32_t kOverlayPadY = 1u;
constexpr VkDeviceSize kOverlayBufferBytes = (4u + kOverlayMaxCols * kOverlayMaxRows) * sizeof(uint32_t);
}

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
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &outBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device, outBuf, &mr);
    uint32_t typeIndex = findMemoryType(phys, mr.memoryTypeBits, flags);
    if (typeIndex == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
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
                             VkBufferUsageFlags usage) {
    VkDeviceSize allocSize = std::max<VkDeviceSize>(requiredBytes, static_cast<VkDeviceSize>(16));
    VkBufferUsageFlags requiredUsage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (buf.buffer != VK_NULL_HANDLE && buf.capacity >= allocSize && buf.usage == requiredUsage) {
        buf.capacity = std::max(buf.capacity, allocSize);
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

    const size_t bindingCountPerSet = 9; // bindings 3..8 plus material/palette buffers
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

        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
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
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(vk.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

bool Raytracer::createImages(platform::VulkanContext& vk, platform::Swapchain& swap) {
    extent_ = swap.extent();
    currColorFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    motionFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;

    auto createStorageImage = [&](VkFormat fmt, VkImage& image, VkDeviceMemory& memory, VkImageView& view) -> bool {
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { extent_.width, extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk.device(), &ici, nullptr, &image) != VK_SUCCESS) return false;
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(vk.device(), image, &mr);
        uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
        if (vkAllocateMemory(vk.device(), &mai, nullptr, &memory) != VK_SUCCESS) return false;
        vkBindImageMemory(vk.device(), image, memory, 0);
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk.device(), &vci, nullptr, &view) != VK_SUCCESS) return false;
        return true;
    };

    if (!createStorageImage(currColorFormat_, currColorImage_, currColorMem_, currColorView_)) return false;
    if (!createStorageImage(motionFormat_, motionImage_, motionMem_, motionView_)) return false;
    if (!createStorageImage(currColorFormat_, albedoImage_, albedoMem_, albedoView_)) return false;
    if (!createStorageImage(currColorFormat_, normalImage_, normalMem_, normalView_)) return false;
    if (!createStorageImage(currColorFormat_, momentsImage_, momentsMem_, momentsView_)) return false;
    for (size_t i = 0; i < history_.size(); ++i) {
        if (!createStorageImage(currColorFormat_, history_[i].image, history_[i].memory, history_[i].view)) return false;
    }
    for (size_t i = 0; i < historyMoments_.size(); ++i) {
        if (!createStorageImage(currColorFormat_, historyMoments_[i].image, historyMoments_[i].memory, historyMoments_[i].view)) return false;
    }
    historyReadIndex_ = 0;
    historyWriteIndex_ = 1;
    historyInitialized_ = false;

    // Transition all images to GENERAL layout
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = vk.commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb{};
    vkAllocateCommandBuffers(vk.device(), &cbai, &cb);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    std::vector<VkImageMemoryBarrier2> barriers;
    auto addBarrier = [&](VkImage image) {
        VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
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
    addBarrier(currColorImage_);
    addBarrier(motionImage_);
    addBarrier(albedoImage_);
    addBarrier(normalImage_);
    addBarrier(momentsImage_);
    addBarrier(history_[0].image);
    addBarrier(history_[1].image);
    addBarrier(historyMoments_[0].image);
    addBarrier(historyMoments_[1].image);
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cb, &dep);
    vkEndCommandBuffer(cb);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    vkQueueSubmit(vk.graphicsQueue(), 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk.graphicsQueue());
    vkFreeCommandBuffers(vk.device(), vk.commandPool(), 1, &cb);
    return true;
}

void Raytracer::destroyImages(platform::VulkanContext& vk) {
    for (auto& hist : history_) {
        if (hist.view) { vkDestroyImageView(vk.device(), hist.view, nullptr); hist.view = VK_NULL_HANDLE; }
        if (hist.image) { vkDestroyImage(vk.device(), hist.image, nullptr); hist.image = VK_NULL_HANDLE; }
        if (hist.memory) { vkFreeMemory(vk.device(), hist.memory, nullptr); hist.memory = VK_NULL_HANDLE; }
    }
    for (auto& hist : historyMoments_) {
        if (hist.view) { vkDestroyImageView(vk.device(), hist.view, nullptr); hist.view = VK_NULL_HANDLE; }
        if (hist.image) { vkDestroyImage(vk.device(), hist.image, nullptr); hist.image = VK_NULL_HANDLE; }
        if (hist.memory) { vkFreeMemory(vk.device(), hist.memory, nullptr); hist.memory = VK_NULL_HANDLE; }
    }
    if (currColorView_) { vkDestroyImageView(vk.device(), currColorView_, nullptr); currColorView_ = VK_NULL_HANDLE; }
    if (currColorImage_) { vkDestroyImage(vk.device(), currColorImage_, nullptr); currColorImage_ = VK_NULL_HANDLE; }
    if (currColorMem_)   { vkFreeMemory(vk.device(), currColorMem_, nullptr); currColorMem_ = VK_NULL_HANDLE; }
    if (motionView_) { vkDestroyImageView(vk.device(), motionView_, nullptr); motionView_ = VK_NULL_HANDLE; }
    if (motionImage_) { vkDestroyImage(vk.device(), motionImage_, nullptr); motionImage_ = VK_NULL_HANDLE; }
    if (motionMem_)   { vkFreeMemory(vk.device(), motionMem_, nullptr); motionMem_ = VK_NULL_HANDLE; }
    if (albedoView_) { vkDestroyImageView(vk.device(), albedoView_, nullptr); albedoView_ = VK_NULL_HANDLE; }
    if (albedoImage_) { vkDestroyImage(vk.device(), albedoImage_, nullptr); albedoImage_ = VK_NULL_HANDLE; }
    if (albedoMem_) { vkFreeMemory(vk.device(), albedoMem_, nullptr); albedoMem_ = VK_NULL_HANDLE; }
    if (normalView_) { vkDestroyImageView(vk.device(), normalView_, nullptr); normalView_ = VK_NULL_HANDLE; }
    if (normalImage_) { vkDestroyImage(vk.device(), normalImage_, nullptr); normalImage_ = VK_NULL_HANDLE; }
    if (normalMem_) { vkFreeMemory(vk.device(), normalMem_, nullptr); normalMem_ = VK_NULL_HANDLE; }
    if (momentsView_) { vkDestroyImageView(vk.device(), momentsView_, nullptr); momentsView_ = VK_NULL_HANDLE; }
    if (momentsImage_) { vkDestroyImage(vk.device(), momentsImage_, nullptr); momentsImage_ = VK_NULL_HANDLE; }
    if (momentsMem_) { vkFreeMemory(vk.device(), momentsMem_, nullptr); momentsMem_ = VK_NULL_HANDLE; }
    historyReadIndex_ = 0;
    historyWriteIndex_ = 1;
    historyInitialized_ = false;
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
    VkDescriptorSetLayoutBinding bOverlayBuf{}; bOverlayBuf.binding=26; bOverlayBuf.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bOverlayBuf.descriptorCount=1; bOverlayBuf.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOverlayImage{}; bOverlayImage.binding=27; bOverlayImage.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bOverlayImage.descriptorCount=1; bOverlayImage.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[28] = { bUbo, bAcc, bOut, bBH, bOcc, bHK, bHV, bMK, bMV, bDBG, bRayQ, bHitQ, bMissQ, bSecQ, bStats, bMotion, bHistRead, bHistWrite, bAlbedo, bNormal, bMoments, bHistMomentsRead, bHistMomentsWrite, bMatIdx, bMatTable, bPalette, bOverlayBuf, bOverlayImage };
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount=28; dslci.pBindings=bindings;
    if (vkCreateDescriptorSetLayout(vk.device(), &dslci, nullptr, &setLayout_) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount=1; plci.pSetLayouts=&setLayout_;
    if (vkCreatePipelineLayout(vk.device(), &plci, nullptr, &pipeLayout_) != VK_SUCCESS) return false;

    auto readFile = [](const char* path)->std::vector<uint32_t>{ std::vector<uint32_t> out; FILE* f=fopen(path,"rb"); if(!f) return out; fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); out.resize((sz+3)/4); fread(out.data(),1,sz,f); fclose(f); return out; };
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
        if (spv.empty()) { spdlog::error("Failed to load shader %s%s", base.c_str(), rel); return VK_NULL_HANDLE; }
        VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = spv.size()*sizeof(uint32_t); smci.pCode = spv.data();
        VkShaderModule sm{}; vkCreateShaderModule(vk.device(), &smci, nullptr, &sm); return sm;
    };
    VkShaderModule smGen   = loadShader("generate_rays.comp.spv");
    VkShaderModule smShade = loadShader("shade.comp.spv");
    VkShaderModule smTemporal = loadShader("denoise_atrous.comp.spv");
    VkShaderModule smTrav  = loadShader("traverse_bricks.comp.spv");
    VkShaderModule smComp  = loadShader("composite.comp.spv");
    VkShaderModule smOverlay = loadShader("overlay.comp.spv");
    if (!smGen || !smShade || !smTemporal || !smTrav || !smComp || !smOverlay) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.pName = "main"; cpci.layout = pipeLayout_;
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
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = swap.imageCount(); dpci.poolSizeCount = 3; dpci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &descPool_) != VK_SUCCESS) return false;
    sets_.resize(swap.imageCount()); std::vector<VkDescriptorSetLayout> layouts(swap.imageCount(), setLayout_);
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = descPool_; dsai.descriptorSetCount = swap.imageCount(); dsai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(vk.device(), &dsai, sets_.data()) != VK_SUCCESS) return false;

    // Create UBO
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = sizeof(GlobalsUBOData);
    bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (vkCreateBuffer(vk.device(), &bi, nullptr, &ubo_) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), ubo_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &uboMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), ubo_, uboMem_, 0);

    // Create Debug buffer (host visible)
    VkBufferCreateInfo bdbg{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bdbg.size = 128 * sizeof(uint32_t); // expanded for richer diagnostics
    bdbg.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(vk.device(), &bdbg, nullptr, &dbgBuf_) != VK_SUCCESS) return false;
    VkMemoryRequirements mrd{}; vkGetBufferMemoryRequirements(vk.device(), dbgBuf_, &mrd);
    uint32_t typeIndexDbg = findMemoryType(vk.physicalDevice(), mrd.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo maid{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; maid.allocationSize = mrd.size; maid.memoryTypeIndex = typeIndexDbg;
    if (vkAllocateMemory(vk.device(), &maid, nullptr, &dbgMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), dbgBuf_, dbgMem_, 0);

    if (!createQueues(vk)) return false;
    if (!createStatsBuffer(vk)) return false;
    if (!createProfilingResources(vk)) return false;
    VkBufferCreateInfo bover{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bover.size = kOverlayBufferBytes;
    bover.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(vk.device(), &bover, nullptr, &overlayBuf_) != VK_SUCCESS) return false;
    VkMemoryRequirements mro{}; vkGetBufferMemoryRequirements(vk.device(), overlayBuf_, &mro);
    uint32_t typeIndexOverlay = findMemoryType(vk.physicalDevice(), mro.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo maio{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; maio.allocationSize = mro.size; maio.memoryTypeIndex = typeIndexOverlay;
    if (vkAllocateMemory(vk.device(), &maio, nullptr, &overlayMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), overlayBuf_, overlayMem_, 0);
    overlayCapacity_ = mro.size;
    overlayActive_ = false;
    void* overlayInit = nullptr;
    if (vkMapMemory(vk.device(), overlayMem_, 0, overlayCapacity_, 0, &overlayInit) == VK_SUCCESS && overlayInit) {
        std::memset(overlayInit, 0, static_cast<size_t>(overlayCapacity_));
        vkUnmapMemory(vk.device(), overlayMem_);
    }
    if (matIdxBuf_.buffer == VK_NULL_HANDLE) {
        if (!ensureBuffer(vk, matIdxBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
    }
    if (paletteBuf_.buffer == VK_NULL_HANDLE) {
        if (!ensureBuffer(vk, paletteBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
    }
    if (materialTableBuf_.buffer == VK_NULL_HANDLE) {
        if (!ensureBuffer(vk, materialTableBuf_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
    }

    // Write descriptors per swapchain image
    for (uint32_t i=0;i<swap.imageCount();++i) {
        VkDescriptorBufferInfo db{}; db.buffer = ubo_; db.offset=0; db.range = sizeof(GlobalsUBOData);
        VkDescriptorImageInfo diCurr{}; diCurr.imageView = currColorView_; diCurr.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diOut{}; diOut.imageView = swap.imageViews()[i]; diOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diMotion{}; diMotion.imageView = motionView_; diMotion.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistRead{}; diHistRead.imageView = history_[0].view; diHistRead.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistWrite{}; diHistWrite.imageView = history_[1].view; diHistWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diAlbedo{}; diAlbedo.imageView = albedoView_; diAlbedo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diNormal{}; diNormal.imageView = normalView_; diNormal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diMoments{}; diMoments.imageView = momentsView_; diMoments.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistMomentsRead{}; diHistMomentsRead.imageView = historyMoments_[0].view; diHistMomentsRead.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diHistMomentsWrite{}; diHistMomentsWrite.imageView = historyMoments_[1].view; diHistMomentsWrite.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo dbDBG{ dbgBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbRay{ rayQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHit{ hitQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMiss{ missQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbSec{ secondaryQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbStats{ statsBuf_, 0, sizeof(TraversalStatsHost) };
        VkDescriptorBufferInfo dbMatIdx{ matIdxBuf_.buffer, 0, matIdxBuf_.size > 0 ? matIdxBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMatTable{ materialTableBuf_.buffer, 0, materialTableBuf_.size > 0 ? materialTableBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbPalette{ paletteBuf_.buffer, 0, paletteBuf_.size > 0 ? paletteBuf_.size : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbOverlay{ overlayBuf_, 0, overlayCapacity_ ? overlayCapacity_ : VK_WHOLE_SIZE };
        VkDescriptorImageInfo diOverlay{}; diOverlay.imageView = swap.imageViews()[i]; diOverlay.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[22]{};
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
        writes[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[20].dstSet=sets_[i]; writes[20].dstBinding=26; writes[20].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[20].descriptorCount=1; writes[20].pBufferInfo=&dbOverlay;
        writes[21].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[21].dstSet=sets_[i]; writes[21].dstBinding=27; writes[21].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[21].descriptorCount=1; writes[21].pImageInfo=&diOverlay;
        uint32_t writeCount = static_cast<uint32_t>(sizeof(writes) / sizeof(writes[0]));
        vkUpdateDescriptorSets(vk.device(), writeCount, writes, 0, nullptr);
    }
    descriptorsReady_ = true;
    refreshWorldDescriptors(vk);
    return true;
}

void Raytracer::destroyDescriptors(platform::VulkanContext& vk) {
    destroyProfilingResources(vk);
    destroyStatsBuffer(vk);
    destroyQueues(vk);
    if (ubo_) { vkDestroyBuffer(vk.device(), ubo_, nullptr); ubo_ = VK_NULL_HANDLE; }
    if (uboMem_) { vkFreeMemory(vk.device(), uboMem_, nullptr); uboMem_ = VK_NULL_HANDLE; }
    if (dbgBuf_) { vkDestroyBuffer(vk.device(), dbgBuf_, nullptr); dbgBuf_ = VK_NULL_HANDLE; }
    if (dbgMem_) { vkFreeMemory(vk.device(), dbgMem_, nullptr); dbgMem_ = VK_NULL_HANDLE; }
    if (overlayBuf_) { vkDestroyBuffer(vk.device(), overlayBuf_, nullptr); overlayBuf_ = VK_NULL_HANDLE; }
    if (overlayMem_) { vkFreeMemory(vk.device(), overlayMem_, nullptr); overlayMem_ = VK_NULL_HANDLE; overlayCapacity_ = 0; }
    overlayActive_ = false;
    overlayCharsX_ = overlayCharsY_ = 0;
    overlayPixelWidth_ = overlayPixelHeight_ = 0;
    if (descPool_) { vkDestroyDescriptorPool(vk.device(), descPool_, nullptr); descPool_ = VK_NULL_HANDLE; }
    sets_.clear();
    descriptorsReady_ = false;
    historyReadIndex_ = 0;
    historyWriteIndex_ = 1;
    historyInitialized_ = false;
}

bool Raytracer::createWorld(platform::VulkanContext& vk) {
    brickStore_ = std::make_unique<world::BrickStore>();
    auto& store = *brickStore_;
    math::PlanetParams P{ 100.0, 12.0, 100.0, 24.0 };
    noiseParams_.continentFrequency = 0.035f;
    noiseParams_.continentAmplitude = 14.0f;
    noiseParams_.continentOctaves   = 5;
    noiseParams_.detailFrequency    = 0.18f;
    noiseParams_.detailAmplitude    = 3.5f;
    noiseParams_.detailOctaves      = 4;
    noiseParams_.warpFrequency      = 0.25f;
    noiseParams_.warpAmplitude      = 0.7f;
    noiseParams_.caveFrequency      = 0.45f;
    noiseParams_.caveAmplitude      = 5.0f;
    noiseParams_.caveThreshold      = 0.3f;
    worldSeed_ = 1337u;
    store.configure(P, /*voxelSize*/0.5f, /*brickDim*/VOXEL_BRICK_SIZE, noiseParams_, worldSeed_);

    aggregateWorld_ = {};
    spdlog::info("Initial world bricks: 0 (0 MiB occupancy)");
    return true;
}

bool Raytracer::uploadWorld(platform::VulkanContext& vk, const world::CpuWorld& cpu) {
    brickCount_ = static_cast<uint32_t>(cpu.headers.size());
    hashCapacity_ = cpu.hashCapacity;
    macroCapacity_ = cpu.macroCapacity;
    macroDimBricks_ = cpu.macroDimBricks;
    macroKeysHost_ = cpu.macroKeys;
    macroValsHost_ = cpu.macroVals;

    auto uploadBuffer = [&](BufferResource& dst, const auto& src, VkDeviceSize elementSize, const char* label) -> bool {
        VkDeviceSize bytes = static_cast<VkDeviceSize>(src.size()) * elementSize;
        if (!ensureBuffer(vk, dst, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            spdlog::error("uploadWorld: ensure buffer failed for {} ({} bytes)", label, bytes);
            return false;
        }
        dst.size = bytes;
        if (bytes == 0 || src.empty()) {
            return true;
        }
        if (!uploadCtx_.uploadBuffer(static_cast<const void*>(src.data()), bytes, dst.buffer)) {
            spdlog::error("uploadWorld: upload failed for {} ({} bytes)", label, bytes);
            return false;
        }
        return true;
    };

    if (!uploadBuffer(bhBuf_, cpu.headers, sizeof(world::BrickHeader), "headers")) return false;
    if (!uploadBuffer(occBuf_, cpu.occWords, sizeof(uint64_t), "occupancy")) return false;
    if (!uploadBuffer(hkBuf_, cpu.hashKeys, sizeof(uint64_t), "hashKeys")) return false;
    if (!uploadBuffer(hvBuf_, cpu.hashVals, sizeof(uint32_t), "hashVals")) return false;
    if (!uploadBuffer(mkBuf_, cpu.macroKeys, sizeof(uint64_t), "macroKeys")) return false;
    if (!uploadBuffer(mvBuf_, cpu.macroVals, sizeof(uint32_t), "macroVals")) return false;
    if (!uploadBuffer(paletteBuf_, cpu.palettes, sizeof(uint32_t), "palettes")) return false;
    if (!uploadBuffer(matIdxBuf_, cpu.materialIndices, sizeof(uint32_t), "materialIndices")) return false;
    if (!uploadBuffer(materialTableBuf_, cpu.materialTable, sizeof(world::MaterialGpu), "materialTable")) return false;

    materialTableHost_ = cpu.materialTable;
    paletteHost_ = cpu.palettes;
    materialCount_ = static_cast<uint32_t>(cpu.materialTable.size());

    markWorldDescriptorsDirty();
    if (descriptorsReady_) {
        refreshWorldDescriptors(vk);
    }

    return true;
}

namespace {
void buildHash(world::CpuWorld& world);
void buildMacroHash(world::CpuWorld& world, uint32_t macroDimBricks);
}

uint64_t Raytracer::packRegionKey(const glm::ivec3& coord) {
    const uint64_t B = 1ull << 20;
    return (static_cast<uint64_t>(coord.x + static_cast<int>(B)) << 42) |
           (static_cast<uint64_t>(coord.y + static_cast<int>(B)) << 21) |
            static_cast<uint64_t>(coord.z + static_cast<int>(B));
}

bool Raytracer::addRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord, world::CpuWorld&& cpu) {
    uint64_t key = packRegionKey(regionCoord);
    regionWorlds_[key] = std::move(cpu);
    bool rebuilt = rebuildGpuWorld(vk);
    if (rebuilt) {
        historyReadIndex_ = 0;
        historyWriteIndex_ = 1;
        historyInitialized_ = false;
    }
    return rebuilt;
}

bool Raytracer::removeRegion(platform::VulkanContext& vk, const glm::ivec3& regionCoord) {
    uint64_t key = packRegionKey(regionCoord);
    auto it = regionWorlds_.find(key);
    if (it == regionWorlds_.end()) {
        return true; // nothing to remove
    }
    regionWorlds_.erase(it);
    bool rebuilt = rebuildGpuWorld(vk);
    if (rebuilt) {
        historyReadIndex_ = 0;
        historyWriteIndex_ = 1;
        historyInitialized_ = false;
    }
    return rebuilt;
}

bool Raytracer::rebuildGpuWorld(platform::VulkanContext& vk) {
    if (regionWorlds_.empty()) {
        destroyWorld(vk);
        aggregateWorld_ = {};
        return true;
    }

    world::CpuWorld combined;

    size_t totalHeaders = 0;
    size_t totalOcc = 0;
    size_t totalMatIdx = 0;
    size_t totalPalettes = 0;
    for (const auto& entry : regionWorlds_) {
        const world::CpuWorld& cpu = entry.second;
        totalHeaders += cpu.headers.size();
        totalOcc += cpu.occWords.size();
        totalMatIdx += cpu.materialIndices.size();
        totalPalettes += cpu.palettes.size();
        if (combined.materialTable.empty() && !cpu.materialTable.empty()) {
            combined.materialTable = cpu.materialTable;
        }
    }

    combined.headers.reserve(totalHeaders);
    combined.occWords.reserve(totalOcc);
    combined.materialIndices.reserve(totalMatIdx);
    combined.palettes.reserve(totalPalettes);

    for (const auto& entry : regionWorlds_) {
        const world::CpuWorld& cpu = entry.second;
        if (cpu.headers.empty()) {
            continue;
        }

        size_t headerStart = combined.headers.size();
        size_t occStart = combined.occWords.size();
        size_t matStart = combined.materialIndices.size();
        size_t paletteStart = combined.palettes.size();

        combined.headers.insert(combined.headers.end(), cpu.headers.begin(), cpu.headers.end());
        combined.occWords.insert(combined.occWords.end(), cpu.occWords.begin(), cpu.occWords.end());
        combined.materialIndices.insert(combined.materialIndices.end(), cpu.materialIndices.begin(), cpu.materialIndices.end());
        combined.palettes.insert(combined.palettes.end(), cpu.palettes.begin(), cpu.palettes.end());

        uint32_t occOffsetBytes = static_cast<uint32_t>(occStart * sizeof(uint64_t));
        uint32_t matOffsetBytes = static_cast<uint32_t>(matStart * sizeof(uint32_t));
        uint32_t paletteOffsetBytes = static_cast<uint32_t>(paletteStart * sizeof(uint32_t));

        for (size_t i = 0; i < cpu.headers.size(); ++i) {
            world::BrickHeader& h = combined.headers[headerStart + i];
            h.occOffset += occOffsetBytes;
            h.matIdxOffset += matOffsetBytes;
            if (h.paletteOffset != world::kInvalidOffset) {
                h.paletteOffset += paletteOffsetBytes;
            }
        }
    }

    buildHash(combined);
    buildMacroHash(combined, 8);

    if (!uploadWorld(vk, combined)) {
        return false;
    }

    aggregateWorld_ = std::move(combined);
    return true;
}

void Raytracer::destroyWorld(platform::VulkanContext& vk) {
    destroyBuffer(vk, bhBuf_);
    destroyBuffer(vk, occBuf_);
    destroyBuffer(vk, hkBuf_);
    destroyBuffer(vk, hvBuf_);
    destroyBuffer(vk, mkBuf_);
    destroyBuffer(vk, mvBuf_);
    destroyBuffer(vk, paletteBuf_);
    destroyBuffer(vk, matIdxBuf_);
    destroyBuffer(vk, materialTableBuf_);
    macroKeysHost_.clear();
    macroValsHost_.clear();
    paletteHost_.clear();
    materialTableHost_.clear();
    brickCount_ = 0;
    hashCapacity_ = 0;
    macroCapacity_ = 0;
    macroDimBricks_ = 0;
    materialCount_ = 0;
    markWorldDescriptorsDirty();
    aggregateWorld_ = {};
    regionWorlds_.clear();
}

namespace {
constexpr VkDeviceSize kQueueHeaderBytes = sizeof(uint32_t) * 4;
constexpr VkDeviceSize kRayPayloadBytes  = 80;
constexpr VkDeviceSize kHitPayloadBytes  = 96;
constexpr VkDeviceSize kMissPayloadBytes = 80;

void buildHash(world::CpuWorld& world) {
    const size_t n = world.headers.size();
    uint32_t cap = 1u;
    while (cap < n * 2u) cap <<= 1u;
    if (cap < 8u) cap = 8u;
    world.hashKeys.assign(cap, 0ull);
    world.hashVals.assign(cap, 0u);

    auto put = [&](uint64_t key, uint32_t val) {
        const uint32_t mask = cap - 1u;
        uint32_t h = static_cast<uint32_t>(((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) >> 32) & mask;
        for (uint32_t probe = 0; probe < cap; ++probe) {
            uint32_t idx = (h + probe) & mask;
            if (world.hashKeys[idx] == 0ull) {
                world.hashKeys[idx] = key;
                world.hashVals[idx] = val;
                return;
            }
        }
    };

    for (uint32_t i = 0; i < world.headers.size(); ++i) {
        const auto& h = world.headers[i];
        const uint64_t B = 1ull << 20;
        uint64_t key = ((uint64_t)(h.bx + (int)B) << 42) |
                       ((uint64_t)(h.by + (int)B) << 21) |
                        (uint64_t)(h.bz + (int)B);
        put(key, i);
    }
    world.hashCapacity = cap;
}

void buildMacroHash(world::CpuWorld& world, uint32_t macroDimBricks) {
    world.macroDimBricks = macroDimBricks;
    std::vector<uint64_t> unique;
    unique.reserve(world.headers.size());
    const auto divFloor = [](int a, int b) {
        int q = a / b;
        int r = a - q * b;
        if (((a ^ b) < 0) && r != 0) --q;
        return q;
    };
    for (const auto& h : world.headers) {
        int mx = divFloor(h.bx, static_cast<int>(macroDimBricks));
        int my = divFloor(h.by, static_cast<int>(macroDimBricks));
        int mz = divFloor(h.bz, static_cast<int>(macroDimBricks));
        const uint64_t B = 1ull << 20;
        unique.push_back(((uint64_t)(mx + (int)B) << 42) |
                         ((uint64_t)(my + (int)B) << 21) |
                          (uint64_t)(mz + (int)B));
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    uint32_t cap = 1u;
    while (cap < unique.size() * 2u) cap <<= 1u;
    if (cap < 8u) cap = 8u;
    world.macroKeys.assign(cap, 0ull);
    world.macroVals.assign(cap, 0u);
    auto put = [&](uint64_t key, uint32_t val) {
        const uint32_t mask = cap - 1u;
        uint64_t kx = key ^ (key >> 33);
        const uint64_t A = 0xff51afd7ed558ccdULL;
        uint32_t h = static_cast<uint32_t>((kx * A) >> 32) & mask;
        for (uint32_t probe = 0; probe < cap; ++probe) {
            uint32_t idx = (h + probe) & mask;
            if (world.macroKeys[idx] == 0ull) {
                world.macroKeys[idx] = key;
                world.macroVals[idx] = val;
                return;
            }
        }
    };
    for (auto key : unique) put(key, 1u);
    world.macroCapacity = cap;
}

uint32_t nextPow2(uint32_t v) {
    if (v <= 1u) return 1u;
    v -= 1u;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}
}

bool Raytracer::createQueues(platform::VulkanContext& vk) {
    destroyQueues(vk);
    uint64_t pixelCount = uint64_t(extent_.width) * uint64_t(extent_.height);
    queueCapacity_ = nextPow2(static_cast<uint32_t>(std::max<uint64_t>(1ull, pixelCount)));
    auto alloc = [&](VkBuffer& buf, VkDeviceMemory& mem, VkDeviceSize payloadBytes) {
        VkDeviceSize size = kQueueHeaderBytes + payloadBytes * queueCapacity_;
        return allocateBuffer(vk.device(), vk.physicalDevice(), size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              buf, mem);
    };
    if (!alloc(rayQueueBuf_, rayQueueMem_, kRayPayloadBytes)) return false;
    if (!alloc(hitQueueBuf_, hitQueueMem_, kHitPayloadBytes)) return false;
    if (!alloc(missQueueBuf_, missQueueMem_, kMissPayloadBytes)) return false;
    if (!alloc(secondaryQueueBuf_, secondaryQueueMem_, kRayPayloadBytes)) return false;
    return true;
}

void Raytracer::destroyQueues(platform::VulkanContext& vk) {
    if (secondaryQueueBuf_) { vkDestroyBuffer(vk.device(), secondaryQueueBuf_, nullptr); secondaryQueueBuf_ = VK_NULL_HANDLE; }
    if (secondaryQueueMem_) { vkFreeMemory(vk.device(), secondaryQueueMem_, nullptr); secondaryQueueMem_ = VK_NULL_HANDLE; }
    if (missQueueBuf_) { vkDestroyBuffer(vk.device(), missQueueBuf_, nullptr); missQueueBuf_ = VK_NULL_HANDLE; }
    if (missQueueMem_) { vkFreeMemory(vk.device(), missQueueMem_, nullptr); missQueueMem_ = VK_NULL_HANDLE; }
    if (hitQueueBuf_) { vkDestroyBuffer(vk.device(), hitQueueBuf_, nullptr); hitQueueBuf_ = VK_NULL_HANDLE; }
    if (hitQueueMem_) { vkFreeMemory(vk.device(), hitQueueMem_, nullptr); hitQueueMem_ = VK_NULL_HANDLE; }
    if (rayQueueBuf_) { vkDestroyBuffer(vk.device(), rayQueueBuf_, nullptr); rayQueueBuf_ = VK_NULL_HANDLE; }
    if (rayQueueMem_) { vkFreeMemory(vk.device(), rayQueueMem_, nullptr); rayQueueMem_ = VK_NULL_HANDLE; }
    queueCapacity_ = 0u;
}

void Raytracer::writeQueueHeaders(VkCommandBuffer cb) {
    if (queueCapacity_ == 0u) return;
    struct Header { uint32_t head; uint32_t tail; uint32_t capacity; uint32_t dropped; } hdr{0u, 0u, queueCapacity_, 0u};
    vkCmdUpdateBuffer(cb, rayQueueBuf_, 0, sizeof(Header), &hdr);
    vkCmdUpdateBuffer(cb, hitQueueBuf_, 0, sizeof(Header), &hdr);
    vkCmdUpdateBuffer(cb, missQueueBuf_, 0, sizeof(Header), &hdr);
    vkCmdUpdateBuffer(cb, secondaryQueueBuf_, 0, sizeof(Header), &hdr);
}

bool Raytracer::createProfilingResources(platform::VulkanContext& vk) {
    destroyProfilingResources(vk);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vk.physicalDevice(), &props);
    timestampPeriodNs_ = props.limits.timestampPeriod;
    VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
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

bool Raytracer::createStatsBuffer(platform::VulkanContext& vk) {
    destroyStatsBuffer(vk);
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = sizeof(TraversalStatsHost);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk.device(), &bi, nullptr, &statsBuf_) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), statsBuf_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIndex == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &statsMem_) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device(), statsBuf_, statsMem_, 0);
    std::memset(&statsHost_, 0, sizeof(statsHost_));
    spdlog::info("Created traversal stats buffer size={} handle={} mem={}", sizeof(TraversalStatsHost), (void*)statsBuf_, (void*)statsMem_);
    return true;
}

void Raytracer::destroyStatsBuffer(platform::VulkanContext& vk) {
    if (statsBuf_) { vkDestroyBuffer(vk.device(), statsBuf_, nullptr); statsBuf_ = VK_NULL_HANDLE; }
    if (statsMem_) { vkFreeMemory(vk.device(), statsMem_, nullptr); statsMem_ = VK_NULL_HANDLE; }
    std::memset(&statsHost_, 0, sizeof(statsHost_));
}

void Raytracer::updateFrameDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap, uint32_t swapIndex) {
    (void)swap;
    if (!descriptorsReady_) return;
    historyWriteIndex_ = historyReadIndex_ ^ 1u;

    VkDescriptorImageInfo diCurr{ VK_NULL_HANDLE, currColorView_, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diMotion{ VK_NULL_HANDLE, motionView_, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistRead{ VK_NULL_HANDLE, history_[historyReadIndex_].view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistWrite{ VK_NULL_HANDLE, history_[historyWriteIndex_].view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diMoments{ VK_NULL_HANDLE, momentsView_, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistMomentsRead{ VK_NULL_HANDLE, historyMoments_[historyReadIndex_].view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diHistMomentsWrite{ VK_NULL_HANDLE, historyMoments_[historyWriteIndex_].view, VK_IMAGE_LAYOUT_GENERAL };

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
    writes[7].dstBinding = 27;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[7].descriptorCount = 1;
    writes[7].pImageInfo = &diOverlay;

    vkUpdateDescriptorSets(vk.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool Raytracer::init(platform::VulkanContext& vk, platform::Swapchain& swap) {
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
    d.historyValid      = historyInitialized_ ? 1u : 0u;
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
        d.noisePad2          = 0.0f;
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
        d.noisePad2 = 0.0f;
        d.noiseSeed = 0u;
        d.noiseContinentOctaves = d.noiseDetailOctaves = d.noiseCaveOctaves = 0u;
    }
    renderOrigin_ = glm::vec3(d.renderOrigin[0], d.renderOrigin[1], d.renderOrigin[2]);
    void* mapped=nullptr; vkMapMemory(vk.device(), uboMem_, 0, sizeof(GlobalsUBOData), 0, &mapped);
    std::memcpy(mapped, &d, sizeof(GlobalsUBOData));
    vkUnmapMemory(vk.device(), uboMem_);
    currFrameIdx_ = d.frameIdx;
}

void Raytracer::record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex) {
    updateFrameDescriptors(vk, swap, swapIndex);
    writeQueueHeaders(cb);
    if (statsBuf_) {
        vkCmdFillBuffer(cb, statsBuf_, 0, sizeof(TraversalStatsHost), 0);
    }
    VkMemoryBarrier2 hdrBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    hdrBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    hdrBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hdrBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    hdrBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo hdrDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    hdrDep.memoryBarrierCount = 1;
    hdrDep.pMemoryBarriers = &hdrBarrier;
    vkCmdPipelineBarrier2(cb, &hdrDep);
    if (timestampPool_) {
        vkCmdResetQueryPool(cb, timestampPool_, 0, kTimestampCount);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 0);
    }

    // Generate primary rays into queue
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeGenerate_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    uint32_t gx = (extent_.width + 7u)/8u;
    uint32_t gy = (extent_.height + 7u)/8u;
    vkCmdDispatch(cb, gx, gy, 1);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 1);
    }
    VkMemoryBarrier2 genBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    genBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    genBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    genBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    genBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo genDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    genDep.memoryBarrierCount = 1;
    genDep.pMemoryBarriers = &genBarrier;
    vkCmdPipelineBarrier2(cb, &genDep);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 2);
    }

    // Traverse rays into queues
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTraverse_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    vkCmdDispatch(cb, gx, gy, 1);
    // Ensure queue writes are visible to shading pass
    VkMemoryBarrier2 queueBarrier{};
    queueBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    queueBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    queueBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    queueBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    queueBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo queueDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    queueDep.memoryBarrierCount = 1;
    queueDep.pMemoryBarriers = &queueBarrier;
    vkCmdPipelineBarrier2(cb, &queueDep);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 3);
    }

    // Shade hits/misses from queues into current color/motion targets
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeShade_);
    uint32_t shadeGroups = gx * gy; // one workgroup per traverse workgroup (64 threads)
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 4);
    }
    vkCmdDispatch(cb, shadeGroups, 1, 1);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 5);
    }

    // Make current color and motion available to temporal accumulation
    std::array<VkImageMemoryBarrier2, 5> preTemporal{};
    auto makeBarrier = [](VkImage image) {
        VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
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
    preTemporal[0] = makeBarrier(currColorImage_);
    preTemporal[1] = makeBarrier(motionImage_);
    preTemporal[2] = makeBarrier(albedoImage_);
    preTemporal[3] = makeBarrier(normalImage_);
    preTemporal[4] = makeBarrier(momentsImage_);
    VkDependencyInfo preTemporalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preTemporalDep.imageMemoryBarrierCount = static_cast<uint32_t>(preTemporal.size());
    preTemporalDep.pImageMemoryBarriers = preTemporal.data();
    vkCmdPipelineBarrier2(cb, &preTemporalDep);

    // Temporal accumulation into history write target
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTemporal_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    uint32_t tx = (extent_.width + 7u) / 8u;
    uint32_t ty = (extent_.height + 7u) / 8u;
    vkCmdDispatch(cb, tx, ty, 1);

    // Barrier history image for composite pass
    std::array<VkImageMemoryBarrier2, 2> histBarriers{};
    histBarriers[0] = makeBarrier(history_[historyWriteIndex_].image);
    histBarriers[1] = makeBarrier(historyMoments_[historyWriteIndex_].image);
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(histBarriers.size());
    dep.pImageMemoryBarriers = histBarriers.data();
    vkCmdPipelineBarrier2(cb, &dep);

    // Composite: read temporally accumulated history, write swap image already in GENERAL from App
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeComposite_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 6);
    }
    vkCmdDispatch(cb, gx, gy, 1);
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, timestampPool_, 7);
    }

    historyReadIndex_ = historyWriteIndex_;
    historyInitialized_ = true;

    // Debug readback moved to readDebug(), called after submit/present to ensure GPU wrote the data.
}

void Raytracer::recordOverlay(VkCommandBuffer cb, uint32_t swapIndex) {
    if (!overlayActive_ || overlayPixelWidth_ == 0 || overlayPixelHeight_ == 0 || pipeOverlay_ == VK_NULL_HANDLE) {
        return;
    }
    if (swapIndex >= sets_.size()) {
        return;
    }
    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cb, &dep);
    uint32_t groupsX = (overlayPixelWidth_ + 7u) / 8u;
    uint32_t groupsY = (overlayPixelHeight_ + 7u) / 8u;
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
    overlayCharsX_ = 0;
    overlayCharsY_ = 0;
    overlayPixelWidth_ = 0;
    overlayPixelHeight_ = 0;
    overlayActive_ = false;

    if (!overlayBuf_ || !overlayMem_) {
        return;
    }

    const uint32_t maxRows = kOverlayMaxRows;
    const uint32_t maxCols = kOverlayMaxCols;
    uint32_t rows = std::min<uint32_t>(lines.size(), maxRows);
    if (rows == 0u) {
        void* mapped = nullptr;
        if (vkMapMemory(vk.device(), overlayMem_, 0, overlayCapacity_, 0, &mapped) == VK_SUCCESS && mapped) {
            std::memset(mapped, 0, static_cast<size_t>(overlayCapacity_));
            vkUnmapMemory(vk.device(), overlayMem_);
        }
        return;
    }

    std::vector<std::string> sanitized;
    sanitized.reserve(rows);
    uint32_t width = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const std::string& line = lines[i];
        std::string clean;
        clean.reserve(maxCols);
        for (char ch : line) {
            if (clean.size() >= maxCols) break;
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
        width = std::max(width, static_cast<uint32_t>(sanitized.back().size()));
    }

    if (width == 0u) {
        void* mapped = nullptr;
        if (vkMapMemory(vk.device(), overlayMem_, 0, overlayCapacity_, 0, &mapped) == VK_SUCCESS && mapped) {
            std::memset(mapped, 0, static_cast<size_t>(overlayCapacity_));
            vkUnmapMemory(vk.device(), overlayMem_);
        }
        return;
    }

    uint32_t stride = width;
    uint32_t totalCells = stride * rows;
    std::vector<uint32_t> payload(4u + totalCells, 0u);
    payload[0] = width;
    payload[1] = rows;
    payload[2] = stride;
    payload[3] = totalCells;
    for (uint32_t r = 0; r < rows; ++r) {
        const std::string& line = sanitized[r];
        for (uint32_t c = 0; c < stride; ++c) {
            unsigned char ch = (c < line.size()) ? static_cast<unsigned char>(line[c]) : static_cast<unsigned char>(' ');
            payload[4u + r * stride + c] = static_cast<uint32_t>(ch);
        }
    }

    VkDeviceSize bytes = static_cast<VkDeviceSize>(payload.size() * sizeof(uint32_t));
    bytes = std::min<VkDeviceSize>(bytes, overlayCapacity_);
    void* mapped = nullptr;
    bool wrote = false;
    if (vkMapMemory(vk.device(), overlayMem_, 0, overlayCapacity_, 0, &mapped) == VK_SUCCESS && mapped) {
        std::memset(mapped, 0, static_cast<size_t>(overlayCapacity_));
        std::memcpy(mapped, payload.data(), static_cast<size_t>(bytes));
        vkUnmapMemory(vk.device(), overlayMem_);
        wrote = true;
    }

    overlayCharsX_ = wrote ? width : 0u;
    overlayCharsY_ = wrote ? rows : 0u;
    overlayPixelWidth_ = wrote ? width * (kOverlayFontWidth + kOverlayPadX) : 0u;
    overlayPixelHeight_ = wrote ? rows * (kOverlayFontHeight + kOverlayPadY) : 0u;
    overlayActive_ = wrote;
}

void Raytracer::readDebug(platform::VulkanContext& vk, uint32_t frameIdx) {
    // Poll infrequently and debounce identical frames to avoid log spam.
    // Temporarily read back every frame for profiling
    void* p=nullptr; if (vkMapMemory(vk.device(), dbgMem_, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) return;
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

    if (statsBuf_) {
        TraversalStatsHost stats{};
        void* mapped = nullptr;
        VkResult mapRes = vkMapMemory(vk.device(), statsMem_, 0, sizeof(TraversalStatsHost), 0, &mapped);
        if (mapRes == VK_SUCCESS && mapped) {
            std::memcpy(&stats, mapped, sizeof(TraversalStatsHost));
            vkUnmapMemory(vk.device(), statsMem_);
            statsHost_ = stats;
            spdlog::debug("Traverse stats: macroVisited={} macroSkipped={} brickSteps={} microSteps={} hits={}",
                          stats.macroVisited, stats.macroSkipped, stats.brickSteps, stats.microSteps, stats.hitsTotal);
        } else {
            spdlog::warn("Failed to map traversal stats buffer (res={})", int(mapRes));
        }
    }
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
