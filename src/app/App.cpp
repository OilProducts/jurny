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
#include <sstream>
#include <iomanip>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include "math/Spherical.h"
#include "core/FrameGraph.h"
#include "core/Jobs.h"
#include "world/Streaming.h"
#include <atomic>
#include <mutex>
#include <numeric>
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

    core::Jobs jobs;
    jobs.start();
    world::Streaming streaming;
    core::FrameGraph frameGraph;

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
    using GlobalsUBO = render::GlobalsUBOData;

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
    spdlog::debug("Raytracer initialized");
    const float planetRadius = 100.0f;
    if (const auto* store = ray.worldStore()) {
        world::Streaming::Config streamCfg;
        streamCfg.shellInner = planetRadius - 25.0f;
        streamCfg.shellOuter = planetRadius + 75.0f;
        streamCfg.shellInner = 0.0f; // use planet defaults from store
        streamCfg.shellOuter = 0.0f;
        streamCfg.keepRadius = 70.0f;
        streamCfg.loadRadius = 110.0f;
        streamCfg.simRadius  = 80.0f;
        streamCfg.regionDimBricks = 16;
        streamCfg.maxRegionSelectionsPerFrame = 4;
        streamCfg.maxConcurrentGenerations = std::max(2, static_cast<int>(jobs.workerCount()));
        streaming.initialize(*store, streamCfg, &jobs);
        spdlog::debug("Streaming initialized");
    } else {
        spdlog::warn("Streaming not initialized: no brick store available");
    }
    VkSemaphoreCreateInfo sciSem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; VkSemaphore acquireSem{}, finishSem{}; vkCreateSemaphore(vk.device(), &sciSem, nullptr, &acquireSem); vkCreateSemaphore(vk.device(), &sciSem, nullptr, &finishSem);
    auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    uint32_t frameCounter = 0;
    bool useShell = false; // toggle with 'V'
    bool mPrevDown = false; // edge-trigger for 'M'
    uint32_t debugFlags = 8u; // start with macro skip enabled (bit3)
    glm::dvec3 camWorld = glm::dvec3(double(planetRadius) + 5.0, 0.0, 5.0);
    glm::dvec3 prevRenderOrigin = camWorld;
    glm::mat4 prevViewMat(1.0f);
    glm::mat4 prevProjMat(1.0f);
    float yawDeg = 180.0f;
    float pitchDeg = 0.0f;
    bool firstMouse = true;
    double lastX = 0.0, lastY = 0.0;
#if VOXEL_ENABLE_WINDOW
    glfwSetInputMode(window.handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
#endif
    while (!window.shouldClose()) {
        window.poll();
        uint32_t idx = 0;
        VkResult acquireRes = vkAcquireNextImageKHR(vk.device(), swap.handle(), UINT64_MAX, acquireSem, VK_NULL_HANDLE, &idx);
        if (acquireRes != VK_SUCCESS) {
            break;
        }

        frameGraph.beginFrame();

#if VOXEL_ENABLE_WINDOW
        if (glfwGetKey(window.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            frameGraph.endFrame();
            break;
        }

        if (glfwGetKey(window.handle(), GLFW_KEY_V) == GLFW_PRESS) useShell = true;
        if (glfwGetKey(window.handle(), GLFW_KEY_B) == GLFW_PRESS) useShell = false;
        if (glfwGetKey(window.handle(), GLFW_KEY_1) == GLFW_PRESS) { debugFlags = 1u; }
        if (glfwGetKey(window.handle(), GLFW_KEY_2) == GLFW_PRESS) { debugFlags = 2u; }
        if (glfwGetKey(window.handle(), GLFW_KEY_3) == GLFW_PRESS) { debugFlags = 4u; }
        if (glfwGetKey(window.handle(), GLFW_KEY_4) == GLFW_PRESS) { debugFlags = 16u; }
        if (glfwGetKey(window.handle(), GLFW_KEY_0) == GLFW_PRESS) debugFlags = 0u;
        int mState = glfwGetKey(window.handle(), GLFW_KEY_M);
        bool mDown = (mState == GLFW_PRESS);
        if (mDown && !mPrevDown) {
            debugFlags ^= 8u;
            spdlog::info("macro-skip {}", (debugFlags & 8u) ? "ON" : "OFF");
        }
        mPrevDown = mDown;
#endif

        GlobalsUBO data{};
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        const float R = planetRadius;
        glm::vec3 forward;
        glm::vec3 up(0.0f, 0.0f, 1.0f);
#if VOXEL_ENABLE_WINDOW
        double mouseX, mouseY;
        glfwGetCursorPos(window.handle(), &mouseX, &mouseY);
        if (firstMouse) {
            lastX = mouseX;
            lastY = mouseY;
            firstMouse = false;
        }
        double xoffset = mouseX - lastX;
        double yoffset = mouseY - lastY;
        lastX = mouseX;
        lastY = mouseY;
        const float sensitivity = 0.08f;
        yawDeg += static_cast<float>(xoffset) * sensitivity;
        pitchDeg += static_cast<float>(yoffset) * sensitivity;
        pitchDeg = std::clamp(pitchDeg, -89.0f, 89.0f);

        const float yawRad = glm::radians(yawDeg);
        const float pitchRad = glm::radians(pitchDeg);
        forward.x = std::cos(pitchRad) * std::cos(yawRad);
        forward.y = std::cos(pitchRad) * std::sin(yawRad);
        forward.z = std::sin(pitchRad);
        forward = glm::normalize(forward);
        glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
        glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        if (glm::dot(right, right) < 1e-4f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        up = glm::normalize(glm::cross(right, forward));

        float moveSpeed = 25.0f;
        if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window.handle(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            moveSpeed *= 2.0f;
        }
        if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) camWorld += glm::dvec3(forward) * double(moveSpeed * dt);
        if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) camWorld -= glm::dvec3(forward) * double(moveSpeed * dt);
        if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) camWorld -= glm::dvec3(right)   * double(moveSpeed * dt);
        if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) camWorld += glm::dvec3(right)   * double(moveSpeed * dt);
        if (glfwGetKey(window.handle(), GLFW_KEY_SPACE) == GLFW_PRESS) camWorld += glm::dvec3(up) * double(moveSpeed * dt);
        if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camWorld -= glm::dvec3(up) * double(moveSpeed * dt);
#else
        forward = glm::vec3(-1.0f, 0.0f, 0.0f);
#endif

        glm::vec3 camPosF = glm::vec3(camWorld);
        streaming.update(camPosF, frameCounter);
        const auto statsBefore = streaming.stats();
        if ((frameCounter % 120u) == 0u) {
            spdlog::info("Streaming stats: selected={} queued={} building={} ready={}",
                         statsBefore.selectedRegions, statsBefore.queuedRegions, statsBefore.buildingRegions, statsBefore.readyRegions);
        }

        std::vector<world::Streaming::ReadyRegion> readyRegionsFrame;
        world::Streaming::ReadyRegion readyRegion{};
        while (streaming.popReadyRegion(readyRegion)) {
            readyRegionsFrame.push_back(std::move(readyRegion));
        }

        if (!readyRegionsFrame.empty()) {
            size_t uploadedRegions = 0;
            size_t uploadedBricks = 0;
            for (auto& region : readyRegionsFrame) {
                const size_t bricksInRegion = region.bricks.headers.size();
                if (ray.addRegion(vk, region.regionCoord, std::move(region.bricks))) {
                    streaming.markRegionUploaded(region.regionCoord);
                    uploadedRegions++;
                    uploadedBricks += bricksInRegion;
                }
            }
            if (uploadedRegions > 0) {
                spdlog::info("Streaming commit applied: regions={} bricks={} (total resident={})",
                             uploadedRegions,
                             uploadedBricks,
                             ray.brickCount());
            }
        }

        glm::ivec3 evictCoord;
        while (streaming.popEvictedRegion(evictCoord)) {
            if (ray.removeRegion(vk, evictCoord)) {
                streaming.markRegionEvicted(evictCoord);
            }
        }

        glm::mat4 V = glm::lookAt(camPosF, camPosF + forward, up);
        float aspect = float(swap.extent().width) / float(swap.extent().height);
        glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.05f, 2000.0f);
        std::memcpy(data.currView, &V[0][0], sizeof(float) * 16);
        std::memcpy(data.currProj, &P[0][0], sizeof(float) * 16);
        std::memcpy(data.prevView, &prevViewMat[0][0], sizeof(float) * 16);
        std::memcpy(data.prevProj, &prevProjMat[0][0], sizeof(float) * 16);
        data.renderOrigin[0] = camPosF.x;
        data.renderOrigin[1] = camPosF.y;
        data.renderOrigin[2] = camPosF.z;
        data.renderOrigin[3] = 0.0f;
        glm::vec3 originDelta = glm::vec3(prevRenderOrigin - camWorld);
        data.originDeltaPrevToCurr[0] = originDelta.x;
        data.originDeltaPrevToCurr[1] = originDelta.y;
        data.originDeltaPrevToCurr[2] = originDelta.z;
        data.originDeltaPrevToCurr[3] = 0.0f;
        data.voxelSize = 0.5f;
        data.brickSize = 4.0f;
        data.Rin = R - 20.0f;
        data.Rout = R + 80.0f;
        data.Rsea = R;
        data.planetRadius = R;
        data.exposure = 1.0f;
        data.frameIdx = frameCounter;
        data.maxBounces = 0;
        data.width = swap.extent().width;
        data.height = swap.extent().height;
        data.raysPerPixel = 1;
        data.flags = debugFlags;
        void* mapped = nullptr;
        vkMapMemory(vk.device(), uboMem, 0, sizeof(GlobalsUBO), 0, &mapped);
        std::memcpy(mapped, &data, sizeof(GlobalsUBO));
        vkUnmapMemory(vk.device(), uboMem);

        uint32_t currentFrameIdx = data.frameIdx;
        ++frameCounter;
        prevViewMat = V;
        prevProjMat = P;
        prevRenderOrigin = camWorld;

        constexpr size_t kOverlayMaxCols = 64;
        const auto statsOverlay = streaming.stats();
        std::vector<std::string> overlayLines;
        overlayLines.reserve(6);
        float fps = (dt > 1e-4f) ? (1.0f / dt) : 0.0f;
        float dtMs = dt * 1000.0f;
        auto clampLine = [&](std::string line) {
            if (line.size() > kOverlayMaxCols) {
                line.resize(kOverlayMaxCols);
            }
            overlayLines.push_back(std::move(line));
        };
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1)
               << "FPS " << std::setw(6) << fps
               << "  DT " << std::setw(6) << dtMs << "MS";
            clampLine(ss.str());
        }
        {
            auto timings = ray.gpuTimingsMs();
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2)
               << "GPU MS G:" << std::setw(5) << timings[0]
               << " T:" << std::setw(5) << timings[1]
               << " S:" << std::setw(5) << timings[2]
               << " C:" << std::setw(5) << timings[3];
            clampLine(ss.str());
        }
        {
            std::ostringstream ss;
            ss << "BRICKS " << ray.brickCount()
               << " REG " << ray.residentRegionCount()
               << " READY " << statsOverlay.readyRegions
               << " QUE " << statsOverlay.queuedRegions
               << " BUILD " << statsOverlay.buildingRegions;
            clampLine(ss.str());
        }
        {
            double altitude = glm::length(camPosF) - double(planetRadius);
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1)
               << "CAM X " << camPosF.x
               << " Y " << camPosF.y
               << " Z " << camPosF.z
               << " ALT " << altitude;
            clampLine(ss.str());
        }
        ray.updateOverlayHUD(vk, overlayLines);

        if (data.frameIdx < 3) {
            auto invV = glm::inverse(V);
            auto invP = glm::inverse(P);
            auto makeRayCPU = [&](int px, int py) {
                float w = float(swap.extent().width), h = float(swap.extent().height);
                glm::vec2 uv = (glm::vec2(px + 0.5f, py + 0.5f)) / glm::vec2(w, h);
                glm::vec2 ndc = glm::vec2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
                glm::vec4 pView = invP * glm::vec4(ndc, 1.0f, 1.0f);
                pView /= std::max(pView.w, 1e-6f);
                glm::vec3 ro = glm::vec3(invV * glm::vec4(0, 0, 0, 1));
                glm::vec3 dirWorld = glm::vec3(invV * glm::vec4(glm::normalize(glm::vec3(pView)), 0));
                glm::vec3 rd = glm::normalize(dirWorld);
                return std::pair<glm::vec3, glm::vec3>(ro, rd);
            };
            auto [ro0, rd0] = makeRayCPU(swap.extent().width / 2, swap.extent().height / 2);
            auto [ro1, rd1] = makeRayCPU(0, 0);
            float tEnter = 0, tExit = 0;
            bool hit = math::IntersectSphereShell(ro0, rd0, data.Rin, data.Rout, tEnter, tExit);
            spdlog::info("Extent={}x{} Rin={} Rout={} eye=({:.1f},{:.1f},{:.1f})",
                         swap.extent().width, swap.extent().height, data.Rin, data.Rout, camPosF.x, camPosF.y, camPosF.z);
            spdlog::info("Center ray ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f}) hit={} t=[{:.1f},{:.1f}]",
                         ro0.x, ro0.y, ro0.z, rd0.x, rd0.y, rd0.z, hit ? 1 : 0, tEnter, tExit);
            spdlog::info("Corner ray  ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f})",
                         ro1.x, ro1.y, ro1.z, rd1.x, rd1.y, rd1.z);
            float dotDirs = glm::dot(rd0, rd1);
            spdlog::info("dot(center,corner)={:.3f} (expect < 1)", dotDirs);
        }

        render::GlobalsUBOData gd = data;
        if (!useShell) {
            ray.updateGlobals(vk, gd);
        }

        frameGraph.addPass("PrepareSwapImage", [&, idx](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = swap.image(idx);
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,
                                 0, nullptr,
                                 0, nullptr,
                                 1, &barrier);
        });

        if (useShell) {
            frameGraph.addPass("ShellVisualize", [&, idx](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[idx], 0, nullptr);
                uint32_t gx = (swap.extent().width + 7u) / 8u;
                uint32_t gy = (swap.extent().height + 7u) / 8u;
                vkCmdDispatch(cmd, gx, gy, 1);
            });
        } else {
            frameGraph.addPass("Raytrace", [&, idx](VkCommandBuffer cmd) {
                ray.record(vk, swap, cmd, idx);
            });
        }

        frameGraph.addPass("OverlayHUD", [&, idx](VkCommandBuffer cmd) {
            ray.recordOverlay(cmd, idx);
        });

        frameGraph.addPass("TransitionToPresent", [&, idx](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = swap.image(idx);
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 0,
                                 0, nullptr,
                                 0, nullptr,
                                 1, &barrier);
        });

        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &bi);
        frameGraph.execute(cb);
        vkEndCommandBuffer(cb);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquireSem;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &finishSem;
        vkQueueSubmit(vk.graphicsQueue(), 1, &si, VK_NULL_HANDLE);

        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        VkSwapchainKHR sc = swap.handle();
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &finishSem;
        pi.swapchainCount = 1;
        pi.pSwapchains = &sc;
        pi.pImageIndices = &idx;
        vkQueuePresentKHR(vk.graphicsQueue(), &pi);
        vkQueueWaitIdle(vk.graphicsQueue());

        frameGraph.endFrame();

        ray.readDebug(vk, currentFrameIdx);
    }

    vkDestroySemaphore(vk.device(), finishSem, nullptr); vkDestroySemaphore(vk.device(), acquireSem, nullptr);
    streaming.shutdown();
    jobs.stop();
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
