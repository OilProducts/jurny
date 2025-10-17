#include "Raytracer.h"
#include "world/BrickStore.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

namespace render {

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

bool Raytracer::createImages(platform::VulkanContext& vk, platform::Swapchain& swap) {
    extent_ = swap.extent();
    accumFormat_ = swap.format();
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = accumFormat_;
    ici.extent = { extent_.width, extent_.height, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(vk.device(), &ici, nullptr, &accumImage_) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(vk.device(), accumImage_, &mr);
    uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
    if (vkAllocateMemory(vk.device(), &mai, nullptr, &accumMem_) != VK_SUCCESS) return false;
    vkBindImageMemory(vk.device(), accumImage_, accumMem_, 0);
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = accumImage_; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = accumFormat_;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; vci.subresourceRange.levelCount = 1; vci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(vk.device(), &vci, nullptr, &accumView_) != VK_SUCCESS) return false;
    return true;
}

void Raytracer::destroyImages(platform::VulkanContext& vk) {
    if (accumView_) { vkDestroyImageView(vk.device(), accumView_, nullptr); accumView_ = VK_NULL_HANDLE; }
    if (accumImage_) { vkDestroyImage(vk.device(), accumImage_, nullptr); accumImage_ = VK_NULL_HANDLE; }
    if (accumMem_)   { vkFreeMemory(vk.device(), accumMem_, nullptr); accumMem_ = VK_NULL_HANDLE; }
}

bool Raytracer::createPipelines(platform::VulkanContext& vk) {
    // Descriptor set layout: 0=UBO, 1=accum, 2=out, 3=BrickHeaders, 4=Occ, 5=HashKeys, 6=HashVals,
    // 7=MacroKeys, 8=MacroVals, 9=Debug, 10=RayQueue, 11=HitQueue, 12=MissQueue, 13=SecondaryQueue
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
    VkDescriptorSetLayoutBinding bindings[14] = { bUbo, bAcc, bOut, bBH, bOcc, bHK, bHV, bMK, bMV, bDBG, bRayQ, bHitQ, bMissQ, bSecQ };
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount=14; dslci.pBindings=bindings;
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
    VkShaderModule smTrav  = loadShader("traverse_bricks.comp.spv");
    VkShaderModule smComp  = loadShader("composite.comp.spv");
    if (!smGen || !smShade || !smTrav || !smComp) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.pName = "main"; cpci.layout = pipeLayout_;
    ss.module = smGen;   cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeGenerate_) != VK_SUCCESS) return false;
    ss.module = smShade; cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeShade_) != VK_SUCCESS) return false;
    ss.module = smTrav;  cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeTraverse_) != VK_SUCCESS) return false;
    ss.module = smComp;  cpci.stage = ss;
    if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeComposite_) != VK_SUCCESS) return false;
    vkDestroyShaderModule(vk.device(), smGen, nullptr);
    vkDestroyShaderModule(vk.device(), smShade, nullptr);
    vkDestroyShaderModule(vk.device(), smTrav, nullptr);
    vkDestroyShaderModule(vk.device(), smComp, nullptr);
    return true;
}

void Raytracer::destroyPipelines(platform::VulkanContext& vk) {
    if (pipeComposite_) { vkDestroyPipeline(vk.device(), pipeComposite_, nullptr); pipeComposite_ = VK_NULL_HANDLE; }
    if (pipeShade_) { vkDestroyPipeline(vk.device(), pipeShade_, nullptr); pipeShade_ = VK_NULL_HANDLE; }
    if (pipeTraverse_) { vkDestroyPipeline(vk.device(), pipeTraverse_, nullptr); pipeTraverse_ = VK_NULL_HANDLE; }
    if (pipeGenerate_) { vkDestroyPipeline(vk.device(), pipeGenerate_, nullptr); pipeGenerate_ = VK_NULL_HANDLE; }
    if (pipeLayout_) { vkDestroyPipelineLayout(vk.device(), pipeLayout_, nullptr); pipeLayout_ = VK_NULL_HANDLE; }
    if (setLayout_) { vkDestroyDescriptorSetLayout(vk.device(), setLayout_, nullptr); setLayout_ = VK_NULL_HANDLE; }
}

bool Raytracer::createDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap) {
    VkDescriptorPoolSize sizes[5] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   2u * swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  11u * swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1 }
    };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = swap.imageCount(); dpci.poolSizeCount = 5; dpci.pPoolSizes = sizes;
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

    // Write descriptors per swapchain image
    for (uint32_t i=0;i<swap.imageCount();++i) {
        VkDescriptorBufferInfo db{}; db.buffer = ubo_; db.offset=0; db.range = sizeof(GlobalsUBOData);
        VkDescriptorImageInfo diAcc{}; diAcc.imageView = accumView_; diAcc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diOut{}; diOut.imageView = swap.imageViews()[i]; diOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo dbBH{ bhBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbOcc{ occBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHK{ hkBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHV{ hvBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbRay{ rayQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHit{ hitQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMiss{ missQueueBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbSec{ secondaryQueueBuf_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[14]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = sets_[i]; writes[0].dstBinding=0; writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].descriptorCount=1; writes[0].pBufferInfo=&db;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = sets_[i]; writes[1].dstBinding=1; writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].descriptorCount=1; writes[1].pImageInfo=&diAcc;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = sets_[i]; writes[2].dstBinding=2; writes[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[2].descriptorCount=1; writes[2].pImageInfo=&diOut;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[3].dstSet = sets_[i]; writes[3].dstBinding=3; writes[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].descriptorCount=1; writes[3].pBufferInfo=&dbBH;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[4].dstSet = sets_[i]; writes[4].dstBinding=4; writes[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[4].descriptorCount=1; writes[4].pBufferInfo=&dbOcc;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[5].dstSet = sets_[i]; writes[5].dstBinding=5; writes[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[5].descriptorCount=1; writes[5].pBufferInfo=&dbHK;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[6].dstSet = sets_[i]; writes[6].dstBinding=6; writes[6].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[6].descriptorCount=1; writes[6].pBufferInfo=&dbHV;
        VkDescriptorBufferInfo dbMK{ mkBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMV{ mvBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbDBG{ dbgBuf_, 0, VK_WHOLE_SIZE };
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[7].dstSet = sets_[i]; writes[7].dstBinding=7; writes[7].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[7].descriptorCount=1; writes[7].pBufferInfo=&dbMK;
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[8].dstSet = sets_[i]; writes[8].dstBinding=8; writes[8].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[8].descriptorCount=1; writes[8].pBufferInfo=&dbMV;
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[9].dstSet = sets_[i]; writes[9].dstBinding=9; writes[9].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[9].descriptorCount=1; writes[9].pBufferInfo=&dbDBG;
        writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[10].dstSet=sets_[i]; writes[10].dstBinding=10; writes[10].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[10].descriptorCount=1; writes[10].pBufferInfo=&dbRay;
        writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[11].dstSet=sets_[i]; writes[11].dstBinding=11; writes[11].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[11].descriptorCount=1; writes[11].pBufferInfo=&dbHit;
        writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[12].dstSet=sets_[i]; writes[12].dstBinding=12; writes[12].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[12].descriptorCount=1; writes[12].pBufferInfo=&dbMiss;
        writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[13].dstSet=sets_[i]; writes[13].dstBinding=13; writes[13].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[13].descriptorCount=1; writes[13].pBufferInfo=&dbSec;
        vkUpdateDescriptorSets(vk.device(), 14, writes, 0, nullptr);
    }
    return true;
}

void Raytracer::destroyDescriptors(platform::VulkanContext& vk) {
    destroyQueues(vk);
    if (ubo_) { vkDestroyBuffer(vk.device(), ubo_, nullptr); ubo_ = VK_NULL_HANDLE; }
    if (uboMem_) { vkFreeMemory(vk.device(), uboMem_, nullptr); uboMem_ = VK_NULL_HANDLE; }
    if (dbgBuf_) { vkDestroyBuffer(vk.device(), dbgBuf_, nullptr); dbgBuf_ = VK_NULL_HANDLE; }
    if (dbgMem_) { vkFreeMemory(vk.device(), dbgMem_, nullptr); dbgMem_ = VK_NULL_HANDLE; }
    if (descPool_) { vkDestroyDescriptorPool(vk.device(), descPool_, nullptr); descPool_ = VK_NULL_HANDLE; }
    sets_.clear();
}

bool Raytracer::createWorld(platform::VulkanContext& vk) {
    // Generate CPU bricks
    world::BrickStore gen;
    math::PlanetParams P{ 10000.0, 300.0, 10000.0, 2000.0 };
    // Generate a moderate spherical cap near +X (safe size for debug).
    // Tangential extents: ±200 m; radial half-thickness: 16 m.
    gen.generateSphericalCap(P, /*voxelSize*/0.5f, /*brickDim*/VOXEL_BRICK_SIZE,
                             /*yExtent*/200.0f, /*zExtent*/200.0f, /*radialHalfThickness*/16.0f);
    const auto& cpu = gen.cpu();
    brickCount_ = (uint32_t)cpu.headers.size();
    hashCapacity_ = cpu.hashCapacity;
    spdlog::info("World cap: bricks={} hashCap={} occWords={} (~{:.2f} MB)",
                 brickCount_, hashCapacity_, cpu.occWords.size(),
                 double(cpu.occWords.size()*sizeof(uint64_t)) / (1024.0*1024.0));

    if (brickCount_ == 0) {
        spdlog::warn("No bricks generated — overlay will show blue everywhere.");
    } else {
        // Log brick bounds and a few samples
        int bxMin=INT32_MAX, byMin=INT32_MAX, bzMin=INT32_MAX;
        int bxMax=INT32_MIN, byMax=INT32_MIN, bzMax=INT32_MIN;
        for (const auto& h : cpu.headers) {
            bxMin = std::min(bxMin, h.bx); byMin = std::min(byMin, h.by); bzMin = std::min(bzMin, h.bz);
            bxMax = std::max(bxMax, h.bx); byMax = std::max(byMax, h.by); bzMax = std::max(bzMax, h.bz);
        }
        spdlog::info("Brick bounds bx:[{}..{}] by:[{}..{}] bz:[{}..{}] (brickSize={} m)",
                     bxMin,bxMax,byMin,byMax,bzMin,bzMax, float(VOXEL_BRICK_SIZE)*0.5f);
        for (size_t i=0;i<std::min<size_t>(brickCount_, 8); ++i) {
            const auto& h = cpu.headers[i];
            spdlog::info("  h[{}]: bc=({}, {}, {}), occOffset={}B", i, h.bx, h.by, h.bz, h.occOffset);
        }
    }

    // Log macro hash bounds to validate CPU build vs expected ranges
    {
        auto unpackKey = [](uint64_t key, int& x, int& y, int& z){
            const int B = 1<<20;
            x = int((key >> 42) & ((1ull<<21)-1)) - B;
            y = int((key >> 21) & ((1ull<<21)-1)) - B;
            z = int((key >>  0) & ((1ull<<21)-1)) - B;
        };
        int mxMin=INT32_MAX,myMin=INT32_MAX,mzMin=INT32_MAX;
        int mxMax=INT32_MIN,myMax=INT32_MIN,mzMax=INT32_MIN;
        uint32_t presentCount=0;
        for (size_t i=0;i<cpu.macroKeys.size();++i){
            uint64_t k = cpu.macroKeys[i]; if (!k) continue; int x,y,z; unpackKey(k,x,y,z);
            mxMin = std::min(mxMin, x); myMin = std::min(myMin, y); mzMin = std::min(mzMin, z);
            mxMax = std::max(mxMax, x); myMax = std::max(myMax, y); mzMax = std::max(mzMax, z);
            presentCount++;
        }
        spdlog::info("Macro hash: cap={} present={} dimBricks={} bounds mx:[{}..{}] my:[{}..{}] mz:[{}..{}]", cpu.macroCapacity, presentCount, cpu.macroDimBricks, mxMin,mxMax,myMin,myMax,mzMin,mzMax);
    }

    auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem){
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(vk.device(), &bi, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), buf, &mr);
        uint32_t typeIndex = findMemoryType(vk.physicalDevice(), mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
        if (vkAllocateMemory(vk.device(), &mai, nullptr, &mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(vk.device(), buf, mem, 0);
        return true;
    };

    // Upload headers
    if (!createBuffer(cpu.headers.size()*sizeof(world::BrickHeader), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, bhBuf_, bhMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), bhMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.headers.data(), cpu.headers.size()*sizeof(world::BrickHeader)); vkUnmapMemory(vk.device(), bhMem_);}    
    // Upload occupancy
    if (!createBuffer(cpu.occWords.size()*sizeof(uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, occBuf_, occMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), occMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.occWords.data(), cpu.occWords.size()*sizeof(uint64_t)); vkUnmapMemory(vk.device(), occMem_);}    
    // Upload hash
    if (!createBuffer(cpu.hashKeys.size()*sizeof(uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hkBuf_, hkMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), hkMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.hashKeys.data(), cpu.hashKeys.size()*sizeof(uint64_t)); vkUnmapMemory(vk.device(), hkMem_);}    
    if (!createBuffer(cpu.hashVals.size()*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hvBuf_, hvMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), hvMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.hashVals.data(), cpu.hashVals.size()*sizeof(uint32_t)); vkUnmapMemory(vk.device(), hvMem_);}    
    // Upload macro hash
    macroCapacity_ = cpu.macroCapacity; macroDimBricks_ = cpu.macroDimBricks;
    if (!createBuffer(cpu.macroKeys.size()*sizeof(uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mkBuf_, mkMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), mkMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.macroKeys.data(), cpu.macroKeys.size()*sizeof(uint64_t)); vkUnmapMemory(vk.device(), mkMem_);}    
    if (!createBuffer(cpu.macroVals.size()*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mvBuf_, mvMem_)) return false;
    { void* m=nullptr; vkMapMemory(vk.device(), mvMem_, 0, VK_WHOLE_SIZE, 0, &m); std::memcpy(m, cpu.macroVals.data(), cpu.macroVals.size()*sizeof(uint32_t)); vkUnmapMemory(vk.device(), mvMem_);}    
    return true;
}

void Raytracer::destroyWorld(platform::VulkanContext& vk) {
    if (bhBuf_) { vkDestroyBuffer(vk.device(), bhBuf_, nullptr); bhBuf_=VK_NULL_HANDLE; }
    if (bhMem_) { vkFreeMemory(vk.device(), bhMem_, nullptr); bhMem_=VK_NULL_HANDLE; }
    if (occBuf_) { vkDestroyBuffer(vk.device(), occBuf_, nullptr); occBuf_=VK_NULL_HANDLE; }
    if (occMem_) { vkFreeMemory(vk.device(), occMem_, nullptr); occMem_=VK_NULL_HANDLE; }
    if (hkBuf_) { vkDestroyBuffer(vk.device(), hkBuf_, nullptr); hkBuf_=VK_NULL_HANDLE; }
    if (hkMem_) { vkFreeMemory(vk.device(), hkMem_, nullptr); hkMem_=VK_NULL_HANDLE; }
    if (hvBuf_) { vkDestroyBuffer(vk.device(), hvBuf_, nullptr); hvBuf_=VK_NULL_HANDLE; }
    if (hvMem_) { vkFreeMemory(vk.device(), hvMem_, nullptr); hvMem_=VK_NULL_HANDLE; }
    if (mkBuf_) { vkDestroyBuffer(vk.device(), mkBuf_, nullptr); mkBuf_=VK_NULL_HANDLE; }
    if (mkMem_) { vkFreeMemory(vk.device(), mkMem_, nullptr); mkMem_=VK_NULL_HANDLE; }
    if (mvBuf_) { vkDestroyBuffer(vk.device(), mvBuf_, nullptr); mvBuf_=VK_NULL_HANDLE; }
    if (mvMem_) { vkFreeMemory(vk.device(), mvMem_, nullptr); mvMem_=VK_NULL_HANDLE; }
}

namespace {
constexpr VkDeviceSize kQueueHeaderBytes = sizeof(uint32_t) * 4;
constexpr VkDeviceSize kRayPayloadBytes  = 80;
constexpr VkDeviceSize kHitPayloadBytes  = 96;
constexpr VkDeviceSize kMissPayloadBytes = 80;

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

bool Raytracer::init(platform::VulkanContext& vk, platform::Swapchain& swap) {
    if (!createImages(vk, swap)) return false;
    if (!createPipelines(vk)) return false;
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
    void* mapped=nullptr; vkMapMemory(vk.device(), uboMem_, 0, sizeof(GlobalsUBOData), 0, &mapped);
    std::memcpy(mapped, &d, sizeof(GlobalsUBOData));
    vkUnmapMemory(vk.device(), uboMem_);
    currFrameIdx_ = d.frameIdx;
}

void Raytracer::record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex) {
    writeQueueHeaders(cb);
    VkMemoryBarrier2 hdrBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    hdrBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    hdrBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hdrBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    hdrBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo hdrDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    hdrDep.memoryBarrierCount = 1;
    hdrDep.pMemoryBarriers = &hdrBarrier;
    vkCmdPipelineBarrier2(cb, &hdrDep);

    // Generate primary rays into queue
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeGenerate_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    uint32_t gx = (extent_.width + 7u)/8u;
    uint32_t gy = (extent_.height + 7u)/8u;
    vkCmdDispatch(cb, gx, gy, 1);
    VkMemoryBarrier2 genBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    genBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    genBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    genBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    genBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo genDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    genDep.memoryBarrierCount = 1;
    genDep.pMemoryBarriers = &genBarrier;
    vkCmdPipelineBarrier2(cb, &genDep);

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

    // Shade hits/misses from queues into accum image
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeShade_);
    uint32_t shadeGroups = gx * gy; // one workgroup per traverse workgroup (64 threads)
    vkCmdDispatch(cb, shadeGroups, 1, 1);

    // Barrier accum for read by composite and out image for write
    VkImageMemoryBarrier2 barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask= VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask= VK_ACCESS_2_SHADER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL; barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].image = accumImage_;
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; barriers[0].subresourceRange.levelCount=1; barriers[0].subresourceRange.layerCount=1;
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cb, &dep);
    // Composite: read accum, write swap image already in GENERAL from App
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeComposite_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    vkCmdDispatch(cb, gx, gy, 1);

    // Debug readback moved to readDebug(), called after submit/present to ensure GPU wrote the data.
}

void Raytracer::readDebug(platform::VulkanContext& vk, uint32_t frameIdx) {
    // Poll infrequently and debounce identical frames to avoid log spam.
    if ((frameIdx % 30u) != 0u) return;
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
        spdlog::info("DBG frame={} flags={} macroCap={} macroDim={} start-bc=({}, {}, {}) start-mc=({}, {}, {}) present={} msteps={} hit-bc=({}, {}, {}) hit-present={}",
                     frame, flags, mcap, mdim,
                     bcx0,bcy0,bcz0, mcx0,mcy0,mcz0, present0, macroStepCount,
                     bcxH,bcyH,bczH, presentHit);
        spdlog::info("DBG macroDiag: zeroMask={} hugeCnt={} nanCnt={} breakMask={} zeroSentUses={} clampCnt={} lastTNext={:.6f} lastTSearch={:.6f}",
                     zeroDirMask, hugeStepCount, nanCount, breakMask, zeroSentinelUseCount, clampCount, lastTNext, lastTSearch);
        // Decode extended diagnostics (band start and sea-level expectations)
        int mcStartX = int(u[24]), mcStartY = int(u[25]), mcStartZ = int(u[26]); uint32_t presentStart = u[27];
        int mcSeaX   = int(u[28]), mcSeaY   = int(u[29]), mcSeaZ   = int(u[30]); uint32_t presentSea = u[31];
        int bcBandX  = int(u[32]), bcBandY  = int(u[33]), bcBandZ  = int(u[34]);
        int bcSeaX   = int(u[35]), bcSeaY   = int(u[36]), bcSeaZ   = int(u[37]);
        spdlog::info("DBG bandStart: mc=({}, {}, {}) present={} bc=({}, {}, {}) | sea: mc=({}, {}, {}) present={} bc=({}, {}, {})",
                     mcStartX, mcStartY, mcStartZ, presentStart,
                     bcBandX, bcBandY, bcBandZ,
                     mcSeaX, mcSeaY, mcSeaZ, presentSea,
                     bcSeaX, bcSeaY, bcSeaZ);
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
        spdlog::info("DBG times: tEnter={:.3f} tExit={:.3f} s0={:.3f} s1={:.3f} sNear={:.3f} | band=[{:.3f}..{:.3f}] macro=[{:.3f}..{:.3f}]",
                     tEnterF, tExitF, s0F, s1F, sNearF, tBandMinF, tBandMaxF, tMacroMinF, tMacroMaxF);
        lastDbgFrame_ = frame; lastMcX_=mcx0; lastMcY_=mcy0; lastMcZ_=mcz0; lastPresent_=int(present0);
    }
    vkUnmapMemory(vk.device(), dbgMem_);
}

void Raytracer::shutdown(platform::VulkanContext& vk) {
    destroyDescriptors(vk);
    destroyPipelines(vk);
    destroyImages(vk);
    destroyWorld(vk);
}

}
