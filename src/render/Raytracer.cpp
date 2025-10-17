#include "Raytracer.h"
#include "world/BrickStore.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

namespace render {

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i) {
        if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
    }
    return UINT32_MAX;
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
    // Descriptor set layout: 0=UBO, 1=accum, 2=out, 3=BrickHeaders, 4=Occ, 5=HashKeys, 6=HashVals, 7=MacroKeys, 8=MacroVals
    VkDescriptorSetLayoutBinding bUbo{ }; bUbo.binding=0; bUbo.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; bUbo.descriptorCount=1; bUbo.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bAcc{ }; bAcc.binding=1; bAcc.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bAcc.descriptorCount=1; bAcc.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOut{ }; bOut.binding=2; bOut.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bOut.descriptorCount=1; bOut.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bBH{ };  bBH.binding=3;  bBH.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bBH.descriptorCount=1; bBH.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bOcc{ }; bOcc.binding=4; bOcc.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bOcc.descriptorCount=1; bOcc.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHK{ };  bHK.binding=5;  bHK.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bHK.descriptorCount=1; bHK.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bHV{ };  bHV.binding=6;  bHV.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bHV.descriptorCount=1; bHV.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMK{ };  bMK.binding=7;  bMK.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMK.descriptorCount=1; bMK.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bMV{ };  bMV.binding=8;  bMV.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bMV.descriptorCount=1; bMV.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[9] = { bUbo, bAcc, bOut, bBH, bOcc, bHK, bHV, bMK, bMV };
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount=9; dslci.pBindings=bindings;
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
    VkShaderModule smShade = loadShader("shade.comp.spv");
    VkShaderModule smTrav  = loadShader("traverse_bricks.comp.spv");
    VkShaderModule smComp  = loadShader("composite.comp.spv");
    if (!smShade || !smTrav || !smComp) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.pName = "main"; cpci.layout = pipeLayout_;
    ss.module = smShade; cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeShade_) != VK_SUCCESS) return false;
    ss.module = smTrav;  cpci.stage = ss; if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeTraverse_) != VK_SUCCESS) return false;
    ss.module = smComp;  cpci.stage = ss;
    if (vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeComposite_) != VK_SUCCESS) return false;
    vkDestroyShaderModule(vk.device(), smShade, nullptr);
    vkDestroyShaderModule(vk.device(), smTrav, nullptr);
    vkDestroyShaderModule(vk.device(), smComp, nullptr);
    return true;
}

void Raytracer::destroyPipelines(platform::VulkanContext& vk) {
    if (pipeComposite_) { vkDestroyPipeline(vk.device(), pipeComposite_, nullptr); pipeComposite_ = VK_NULL_HANDLE; }
    if (pipeShade_) { vkDestroyPipeline(vk.device(), pipeShade_, nullptr); pipeShade_ = VK_NULL_HANDLE; }
    if (pipeLayout_) { vkDestroyPipelineLayout(vk.device(), pipeLayout_, nullptr); pipeLayout_ = VK_NULL_HANDLE; }
    if (setLayout_) { vkDestroyDescriptorSetLayout(vk.device(), setLayout_, nullptr); setLayout_ = VK_NULL_HANDLE; }
}

bool Raytracer::createDescriptors(platform::VulkanContext& vk, platform::Swapchain& swap) {
    VkDescriptorPoolSize sizes[4] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   2u * swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  6u * swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1 }
    };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = swap.imageCount(); dpci.poolSizeCount = 4; dpci.pPoolSizes = sizes;
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

    // Write descriptors per swapchain image
    for (uint32_t i=0;i<swap.imageCount();++i) {
        VkDescriptorBufferInfo db{}; db.buffer = ubo_; db.offset=0; db.range = sizeof(GlobalsUBOData);
        VkDescriptorImageInfo diAcc{}; diAcc.imageView = accumView_; diAcc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo diOut{}; diOut.imageView = swap.imageViews()[i]; diOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo dbBH{ bhBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbOcc{ occBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHK{ hkBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbHV{ hvBuf_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[9]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = sets_[i]; writes[0].dstBinding=0; writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].descriptorCount=1; writes[0].pBufferInfo=&db;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = sets_[i]; writes[1].dstBinding=1; writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].descriptorCount=1; writes[1].pImageInfo=&diAcc;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = sets_[i]; writes[2].dstBinding=2; writes[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[2].descriptorCount=1; writes[2].pImageInfo=&diOut;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[3].dstSet = sets_[i]; writes[3].dstBinding=3; writes[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].descriptorCount=1; writes[3].pBufferInfo=&dbBH;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[4].dstSet = sets_[i]; writes[4].dstBinding=4; writes[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[4].descriptorCount=1; writes[4].pBufferInfo=&dbOcc;
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[5].dstSet = sets_[i]; writes[5].dstBinding=5; writes[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[5].descriptorCount=1; writes[5].pBufferInfo=&dbHK;
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[6].dstSet = sets_[i]; writes[6].dstBinding=6; writes[6].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[6].descriptorCount=1; writes[6].pBufferInfo=&dbHV;
        VkDescriptorBufferInfo dbMK{ mkBuf_, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dbMV{ mvBuf_, 0, VK_WHOLE_SIZE };
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[7].dstSet = sets_[i]; writes[7].dstBinding=7; writes[7].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[7].descriptorCount=1; writes[7].pBufferInfo=&dbMK;
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[8].dstSet = sets_[i]; writes[8].dstBinding=8; writes[8].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[8].descriptorCount=1; writes[8].pBufferInfo=&dbMV;
        vkUpdateDescriptorSets(vk.device(), 9, writes, 0, nullptr);
    }
    return true;
}

void Raytracer::destroyDescriptors(platform::VulkanContext& vk) {
    if (ubo_) { vkDestroyBuffer(vk.device(), ubo_, nullptr); ubo_ = VK_NULL_HANDLE; }
    if (uboMem_) { vkFreeMemory(vk.device(), uboMem_, nullptr); uboMem_ = VK_NULL_HANDLE; }
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
}

void Raytracer::record(platform::VulkanContext& vk, platform::Swapchain& swap, VkCommandBuffer cb, uint32_t swapIndex) {
    // Traverse+shade into accum
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTraverse_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[swapIndex], 0, nullptr);
    uint32_t gx = (extent_.width + 7u)/8u, gy = (extent_.height + 7u)/8u;
    vkCmdDispatch(cb, gx, gy, 1);
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
}

void Raytracer::shutdown(platform::VulkanContext& vk) {
    destroyDescriptors(vk);
    destroyPipelines(vk);
    destroyImages(vk);
    destroyWorld(vk);
}

}
