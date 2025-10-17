#include "App.h"
#include "platform/VulkanContext.h"
#if VOXEL_ENABLE_WINDOW
#include "platform/Window.h"
#include "platform/Swapchain.h"
#include "render/Raytracer.h"
#endif

#include <volk.h>
#include <vector>
#include <array>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include "math/Spherical.h"
#if VOXEL_ENABLE_WINDOW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace app {
int App::run() {
    platform::VulkanContext vk;
    bool enableValidation = true;
#if VOXEL_ENABLE_WINDOW
    // Logging is initialized in main(); no per-frame flush here.
    platform::Window window;
    if (!window.create()) return 1;
    std::vector<const char*> instanceExts;
    platform::Window::getRequiredInstanceExtensions(instanceExts);
    if (!vk.initInstance(instanceExts, enableValidation)) return 1;
    VkSurfaceKHR surface = (VkSurfaceKHR)0;
    if (!window.createSurface(vk.instance(), &surface)) return 1;
    if (!vk.initDevice(surface)) return 1;
    platform::Swapchain swap;
    platform::SwapchainCreateInfo sci{};
    sci.device = vk.device();
    sci.physicalDevice = vk.physicalDevice();
    sci.surface = surface;
    sci.graphicsQueueFamily = vk.graphicsFamily();
    sci.presentQueueFamily = vk.graphicsFamily();
    sci.width = window.width();
    sci.height= window.height();
    swap.create(sci);

    // Shell-visualization compute pipeline (UBO + storage image)
    VkDescriptorSetLayoutBinding bUbo{}; bUbo.binding = 0; bUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; bUbo.descriptorCount = 1; bUbo.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bImg{}; bImg.binding = 1; bImg.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bImg.descriptorCount = 1; bImg.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[2] = { bUbo, bImg };
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 2; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl{}; vkCreateDescriptorSetLayout(vk.device(), &dslci, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl{}; vkCreatePipelineLayout(vk.device(), &plci, nullptr, &pl);

    auto readFile = [](const char* path) -> std::vector<uint32_t> {
        std::vector<uint32_t> out; FILE* f = fopen(path, "rb"); if (!f) return out; fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); out.resize((sz+3)/4); fread(out.data(), 1, sz, f); fclose(f); return out; };
    const char* envAssets = std::getenv("VOXEL_ASSETS_DIR");
    const char* assetsDir = envAssets ? envAssets :
#ifdef VOXEL_ASSETS_DIR
        VOXEL_ASSETS_DIR;
#else
        "assets";
#endif
    std::string spvPath = std::string(assetsDir) + "/shaders/shell_visual.comp.spv";
    std::vector<uint32_t> spv = readFile(spvPath.c_str());
    if (spv.empty()) { spdlog::error("Failed to load %s. You may need to build shaders.", spvPath.c_str()); return 1; }
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size()*sizeof(uint32_t); smci.pCode = spv.data();
    VkShaderModule sm{}; vkCreateShaderModule(vk.device(), &smci, nullptr, &sm);
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ssci{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = sm; ssci.pName = "main"; cpci.stage = ssci; cpci.layout = pl;
    VkPipeline pipe{}; vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);

    // Create a small UBO
    struct alignas(16) GlobalsUBO {
        float currView[16];
        float currProj[16];
        float prevView[16];
        float prevProj[16];
        float originDeltaPrevToCurr[4];
        float voxelSize, brickSize, Rin, Rout;
        float Rsea, exposure; uint32_t frameIdx, maxBounces;
        uint32_t width, height, raysPerPixel, flags;
    };

    VkBuffer ubo{}; VkDeviceMemory uboMem{};
    {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size = sizeof(GlobalsUBO);
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(vk.device(), &bi, nullptr, &ubo);
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), ubo, &mr);
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(vk.physicalDevice(), &mp);
        uint32_t typeIndex = UINT32_MAX;
        for (uint32_t i=0;i<mp.memoryTypeCount;++i) {
            if ((mr.memoryTypeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { typeIndex = i; break; }
        }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
        vkAllocateMemory(vk.device(), &mai, nullptr, &uboMem);
        vkBindBufferMemory(vk.device(), ubo, uboMem, 0);
    }

    // Descriptor pool for UBO + storage images
    VkDescriptorPoolSize poolSizes[2] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  swap.imageCount() },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   swap.imageCount() }
    };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = swap.imageCount(); dpci.poolSizeCount = 2; dpci.pPoolSizes = poolSizes;
    VkDescriptorPool dpool{}; vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &dpool);
    std::vector<VkDescriptorSet> sets(swap.imageCount()); std::vector<VkDescriptorSetLayout> layouts(swap.imageCount(), dsl);
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = swap.imageCount(); dsai.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(vk.device(), &dsai, sets.data());
    for (uint32_t i=0;i<swap.imageCount();++i) {
        VkDescriptorBufferInfo db{}; db.buffer = ubo; db.offset = 0; db.range = sizeof(GlobalsUBO);
        VkDescriptorImageInfo di{}; di.imageView = swap.imageViews()[i]; di.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = sets[i]; writes[0].dstBinding=0; writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].descriptorCount=1; writes[0].pBufferInfo=&db;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = sets[i]; writes[1].dstBinding=1; writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].descriptorCount=1; writes[1].pImageInfo=&di;
        vkUpdateDescriptorSets(vk.device(), 2, writes, 0, nullptr);
    }

    VkCommandBuffer cb{}; VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; cbai.commandPool = vk.commandPool(); cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1; vkAllocateCommandBuffers(vk.device(), &cbai, &cb);

    // Initialize Raytracer M1 skeleton
    render::Raytracer ray;
    if (!ray.init(vk, swap)) {
        spdlog::error("Raytracer init failed.");
        return 1;
    }
    VkSemaphoreCreateInfo sciSem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; VkSemaphore acquireSem{}, finishSem{}; vkCreateSemaphore(vk.device(), &sciSem, nullptr, &acquireSem); vkCreateSemaphore(vk.device(), &sciSem, nullptr, &finishSem);
    auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    uint32_t frameCounter = 0;
    bool useShell = false; // toggle with 'V'
    bool mPrevDown = false; // edge-trigger for 'M'
    uint32_t debugFlags = 8u; // start with macro skip enabled (bit3)
    while (!window.shouldClose()) {
        window.poll(); uint32_t idx=0; if (vkAcquireNextImageKHR(vk.device(), swap.handle(), UINT64_MAX, acquireSem, VK_NULL_HANDLE, &idx) != VK_SUCCESS) break;
#if VOXEL_ENABLE_WINDOW
        if (glfwGetKey(window.handle(), GLFW_KEY_V) == GLFW_PRESS) useShell = true;
        if (glfwGetKey(window.handle(), GLFW_KEY_B) == GLFW_PRESS) useShell = false;
        if (glfwGetKey(window.handle(), GLFW_KEY_1) == GLFW_PRESS) { debugFlags = 1u; }   // probe overlay (red)
        if (glfwGetKey(window.handle(), GLFW_KEY_2) == GLFW_PRESS) { debugFlags = 2u; }   // coarse DDA overlay (yellow)
        if (glfwGetKey(window.handle(), GLFW_KEY_3) == GLFW_PRESS) { debugFlags = 4u; }   // shade at first brick hit
        if (glfwGetKey(window.handle(), GLFW_KEY_4) == GLFW_PRESS) { debugFlags = 16u; }  // macro presence overlay
        int mState = glfwGetKey(window.handle(), GLFW_KEY_M);
        bool mDown = (mState == GLFW_PRESS);
        if (mDown && !mPrevDown) { debugFlags ^= 8u; spdlog::info("macro-skip {}", (debugFlags & 8u) ? "ON" : "OFF"); }
        mPrevDown = mDown;
        if (glfwGetKey(window.handle(), GLFW_KEY_0) == GLFW_PRESS) debugFlags = 0u;   // disable debug overlays
#endif
        // Update UBO (orbit camera for eyeballing)
        GlobalsUBO data{};
        auto now = std::chrono::steady_clock::now();
        float t = std::chrono::duration<float>(now - t0).count();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        float R = 10000.0f; // planet radius
        static float dist = R + 10000.0f; // start further back for clarity
#if VOXEL_ENABLE_WINDOW
        // Simple manual zoom with +/- keys
        if (glfwGetKey(window.handle(), GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window.handle(), GLFW_KEY_KP_ADD) == GLFW_PRESS) dist += 2000.0f * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window.handle(), GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) dist -= 2000.0f * dt;
        dist = std::max(dist, R + 200.0f);
#endif
        float cx = dist * std::cos(0.15f * t);
        float cy = dist * std::sin(0.15f * t);
        float cz = 1500.0f;
        glm::vec3 eyePos(cx,cy,cz);
        glm::mat4 V = glm::lookAt(eyePos, glm::vec3(0,0,0), glm::vec3(0,0,1));
        float aspect = float(swap.extent().width)/float(swap.extent().height);
        glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100000.0f);
        std::memcpy(data.currView, &V[0][0], sizeof(float)*16);
        std::memcpy(data.currProj, &P[0][0], sizeof(float)*16);
        std::memcpy(data.prevView, &V[0][0], sizeof(float)*16);
        std::memcpy(data.prevProj, &P[0][0], sizeof(float)*16);
        data.originDeltaPrevToCurr[0]=0; data.originDeltaPrevToCurr[1]=0; data.originDeltaPrevToCurr[2]=0; data.originDeltaPrevToCurr[3]=0;
        data.voxelSize = 0.5f; data.brickSize = 4.0f; data.Rin = R - 1000.0f; data.Rout = R + 6000.0f;
        data.Rsea = R; data.exposure = 1.0f; data.frameIdx = frameCounter++; data.maxBounces = 0;
        data.width = swap.extent().width; data.height = swap.extent().height; data.raysPerPixel = 1; data.flags = debugFlags;
        void* mapped=nullptr; vkMapMemory(vk.device(), uboMem, 0, sizeof(GlobalsUBO), 0, &mapped);
        std::memcpy(mapped, &data, sizeof(GlobalsUBO));
        vkUnmapMemory(vk.device(), uboMem);

        // One-time debug logging for the center and a corner ray (first few frames only)
        if (frameCounter < 3) {
            auto invV = glm::inverse(V);
            auto invP = glm::inverse(P);
            auto makeRayCPU = [&](int px, int py){
                float w = float(swap.extent().width), h = float(swap.extent().height);
                glm::vec2 uv = (glm::vec2(px + 0.5f, py + 0.5f)) / glm::vec2(w,h);
                glm::vec2 ndc = glm::vec2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
                glm::vec4 pView = invP * glm::vec4(ndc, 1.0f, 1.0f);
                pView /= std::max(pView.w, 1e-6f);
                glm::vec3 ro = glm::vec3(invV * glm::vec4(0,0,0,1));
                glm::vec3 dirWorld = glm::vec3(invV * glm::vec4(glm::normalize(glm::vec3(pView)), 0));
                glm::vec3 rd = glm::normalize(dirWorld);
                return std::pair<glm::vec3, glm::vec3>(ro, rd);
            };
            auto [ro0, rd0] = makeRayCPU(swap.extent().width/2, swap.extent().height/2);
            auto [ro1, rd1] = makeRayCPU(0, 0);
            float tEnter=0, tExit=0;
            bool hit = math::IntersectSphereShell(ro0, rd0, data.Rin, data.Rout, tEnter, tExit);
            spdlog::info("Extent={}x{} Rin={} Rout={} eye=({:.1f},{:.1f},{:.1f})",
                         swap.extent().width, swap.extent().height, data.Rin, data.Rout, eyePos.x, eyePos.y, eyePos.z);
            spdlog::info("Center ray ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f}) hit={} t=[{:.1f},{:.1f}]",
                         ro0.x, ro0.y, ro0.z, rd0.x, rd0.y, rd0.z, hit?1:0, tEnter, tExit);
            spdlog::info("Corner ray  ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f})",
                         ro1.x, ro1.y, ro1.z, rd1.x, rd1.y, rd1.z);
            float dotDirs = glm::dot(rd0, rd1);
            spdlog::info("dot(center,corner)={:.3f} (expect < 1)", dotDirs);
        }

        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; vkBeginCommandBuffer(cb, &bi);
        VkImageMemoryBarrier toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toGeneral.srcAccessMask = 0; toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; toGeneral.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL; toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toGeneral.image = swap.image(idx); toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; toGeneral.subresourceRange.levelCount = 1; toGeneral.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,nullptr, 0,nullptr, 1, &toGeneral);
        if (useShell) {
            // Dispatch the shell visualize pipeline (UBO already updated above)
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[idx], 0, nullptr);
            uint32_t gx = (swap.extent().width + 7u)/8u, gy = (swap.extent().height + 7u)/8u; vkCmdDispatch(cb, gx, gy, 1);
        } else {
            // Use Raytracer skeleton (shade sky -> composite)
            render::GlobalsUBOData gd{};
            std::memcpy(&gd, &data, sizeof(GlobalsUBO));
            ray.updateGlobals(vk, gd);
            ray.record(vk, swap, cb, idx);
        }
        VkImageMemoryBarrier toPresent{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toPresent.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; toPresent.dstAccessMask = 0; toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL; toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toPresent.image = swap.image(idx); toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; toPresent.subresourceRange.levelCount = 1; toPresent.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,nullptr, 0,nullptr, 1, &toPresent);
        vkEndCommandBuffer(cb);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.waitSemaphoreCount=1; si.pWaitSemaphores=&acquireSem; si.pWaitDstStageMask=&waitStage; si.commandBufferCount=1; si.pCommandBuffers=&cb; si.signalSemaphoreCount=1; si.pSignalSemaphores=&finishSem; vkQueueSubmit(vk.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR }; VkSwapchainKHR sc=swap.handle(); pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&finishSem; pi.swapchainCount=1; pi.pSwapchains=&sc; pi.pImageIndices=&idx; vkQueuePresentKHR(vk.graphicsQueue(), &pi); vkQueueWaitIdle(vk.graphicsQueue());
        // After GPU work finishes, read debug buffer (every ~30 frames)
        ray.readDebug(vk, data.frameIdx);
    }

    vkDestroySemaphore(vk.device(), finishSem, nullptr); vkDestroySemaphore(vk.device(), acquireSem, nullptr);
    ray.shutdown(vk);
    vkDestroyPipeline(vk.device(), pipe, nullptr); vkDestroyShaderModule(vk.device(), sm, nullptr);
    vkDestroyDescriptorPool(vk.device(), dpool, nullptr); vkDestroyPipelineLayout(vk.device(), pl, nullptr); vkDestroyDescriptorSetLayout(vk.device(), dsl, nullptr);
    vkDestroyBuffer(vk.device(), ubo, nullptr); vkFreeMemory(vk.device(), uboMem, nullptr);
    swap.destroy(vk.device());
    vk.shutdown();
    window.destroy();
#else
    // Headless: no window, no surface, no swapchain
    if (!vk.initInstance({}, enableValidation)) return 1;
    if (!vk.initDevice(VK_NULL_HANDLE)) return 1; // device without swapchain extension
    // Do a trivial no-op to ensure init path runs
    vk.shutdown();
#endif
    return 0;
}
}
