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
#include <utility>
#include <sstream>
#include <iomanip>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include "math/Spherical.h"
#include "core/FrameGraph.h"
#include "core/Jobs.h"
#include "core/Assets.h"
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
    spdlog::info("Creating window");
    if (!window.create()) {
        spdlog::error("Window creation failed");
        return 1;
    }
    spdlog::info("Window created ({}x{})", window.width(), window.height());
    std::vector<const char*> instanceExts;
    platform::Window::getRequiredInstanceExtensions(instanceExts);
    spdlog::info("Initializing Vulkan instance");
    if (!vk.initInstance(instanceExts, enableValidation)) {
        spdlog::error("initInstance failed");
        return 1;
    }
    spdlog::info("Vulkan instance ready");
    VkSurfaceKHR surface = (VkSurfaceKHR)0;
    if (!window.createSurface(vk.instance(), &surface)) {
        spdlog::error("createSurface failed");
        return 1;
    }
    spdlog::info("Surface created");
    if (!vk.initDevice(surface)) {
        spdlog::error("initDevice failed");
        return 1;
    }
    spdlog::info("Device initialized");
    platform::Swapchain swap;
    platform::SwapchainCreateInfo sci{};
    sci.device = vk.device();
    sci.physicalDevice = vk.physicalDevice();
    sci.surface = surface;
    sci.graphicsQueueFamily = vk.graphicsFamily();
    sci.presentQueueFamily = vk.graphicsFamily();
    sci.width = window.width();
    sci.height= window.height();
    spdlog::info("Creating swapchain");
    if (!swap.create(sci)) {
        spdlog::error("Swapchain creation failed");
        return 1;
    }
    spdlog::info("Swapchain ready");

    core::Jobs jobs;
    jobs.start();
    world::Streaming streaming;
    core::FrameGraph frameGraph;

    const char* envAssets = std::getenv("VOXEL_ASSETS_DIR");
    const char* assetsDir = envAssets ? envAssets :
#ifdef VOXEL_ASSETS_DIR
        VOXEL_ASSETS_DIR;
#else
        "assets";
#endif
    std::string assetRoot = assetsDir ? assetsDir : std::string{};
    core::AssetRegistry assetRegistry;
    bool assetsReady = false;
    if (!assetRoot.empty()) {
        if (assetRegistry.initialize(assetRoot)) {
            assetsReady = true;
            spdlog::info("Asset pack loaded: {} entries", assetRegistry.assetList().size());
            if (!assetRegistry.contains("materials.json")) {
                spdlog::warn("materials.json not found in asset pack");
            }
        } else {
            spdlog::warn("Failed to initialize asset registry at '{}'", assetRoot);
        }
    } else {
        spdlog::warn("Asset directory not specified; proceeding without packed assets");
    }

    VkCommandBuffer cb{};
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk.commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device(), &cbai, &cb);

    // Initialize Raytracer M1 skeleton
    render::Raytracer ray;
    if (assetsReady) {
        ray.setAssetRegistry(&assetRegistry);
    }
    if (!ray.init(vk, swap)) {
        spdlog::error("Raytracer init failed.");
        return 1;
    }
    spdlog::debug("Raytracer initialized");
    const float planetRadius = 100.0f;
    const float eyeHeight = 1.7f;
    const bool enableStreaming = true;
    const world::BrickStore* brickStore = ray.worldStore();
    if (enableStreaming && brickStore) {
        world::Streaming::Config streamCfg;
        streamCfg.shellInner = planetRadius - 25.0f;
        streamCfg.shellOuter = planetRadius + 75.0f;
        streamCfg.shellInner = 0.0f; // use planet defaults from store
        streamCfg.shellOuter = 0.0f;
        streamCfg.keepRadius = 60.0f;
        streamCfg.loadRadius = 90.0f;
        streamCfg.simRadius  = 70.0f;
        streamCfg.regionDimBricks = 8;
        streamCfg.maxRegionSelectionsPerFrame = 2;
        streamCfg.maxConcurrentGenerations = std::max(2, static_cast<int>(jobs.workerCount()));
        streaming.initialize(*brickStore, streamCfg, &jobs);
        spdlog::debug("Streaming initialized");
    } else if (!enableStreaming) {
        spdlog::info("Streaming disabled for debugging");
    } else {
        spdlog::warn("Streaming not initialized: no brick store available");
    }
    VkSemaphoreCreateInfo sciSem{};
    sciSem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore acquireSem{};
    VkSemaphore finishSem{};
    vkCreateSemaphore(vk.device(), &sciSem, nullptr, &acquireSem);
    vkCreateSemaphore(vk.device(), &sciSem, nullptr, &finishSem);
    auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    uint32_t frameCounter = 0;
    bool mPrevDown = false; // edge-trigger for 'M'
    bool blockPrevDown = false; // edge-trigger for 'B'
    const uint32_t blockNormalFlag = 32u;
    uint32_t debugFlags = 8u; // macro skip on, smooth normals by default
    glm::dvec3 camWorld = glm::dvec3(double(planetRadius) + 5.0, 0.0, 5.0);
    glm::dvec3 prevRenderOrigin = camWorld;
    glm::mat4 prevViewMat(1.0f);
    glm::mat4 prevProjMat(1.0f);
    float yawDeg = 180.0f;
    float pitchDeg = 0.0f;
    enum class CameraMode { FreeFly, SurfaceWalk };
    CameraMode cameraMode = CameraMode::FreeFly;
    bool surfaceTogglePrev = false;
    glm::dvec3 walkPos = camWorld;
    bool walkPosValid = false;
    auto sampleSurface = [&](const glm::dvec3& guess) -> std::pair<glm::dvec3, glm::dvec3> {
        if (!brickStore) {
            glm::dvec3 dir = glm::length(guess) > 1e-6 ? glm::normalize(guess) : glm::dvec3(0.0, 0.0, 1.0);
            glm::dvec3 pos = dir * double(planetRadius);
            return {pos, dir};
        }
        glm::vec3 p = glm::vec3(guess);
        const float eps = brickStore->voxelSize();
        glm::vec3 normal = glm::normalize(brickStore->worldGen().crustNormal(p, eps));
        for (int i = 0; i < 8; ++i) {
            float field = brickStore->worldGen().crustField(p);
            if (!std::isfinite(field)) break;
            if (std::abs(field) < 0.01f) break;
            normal = glm::normalize(brickStore->worldGen().crustNormal(p, eps));
            p -= normal * field;
        }
        normal = glm::normalize(brickStore->worldGen().crustNormal(p, eps));
        if (glm::length(normal) < 1e-4f) {
            glm::vec3 dir = glm::length(p) > 1e-6f ? glm::normalize(p) : glm::vec3(0.0f, 0.0f, 1.0f);
            normal = dir;
        }
        return {glm::dvec3(p), glm::dvec3(normal)};
    };
    bool firstMouse = true;
    double lastX = 0.0, lastY = 0.0;
#if VOXEL_ENABLE_WINDOW
    glfwSetInputMode(window.handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
#endif
    size_t debugFrameSerial = 0;
    while (!window.shouldClose()) {
        window.poll();
        uint32_t idx = 0;
        VkResult acquireRes = vkAcquireNextImageKHR(vk.device(), swap.handle(), UINT64_MAX, acquireSem, VK_NULL_HANDLE, &idx);
        if (acquireRes != VK_SUCCESS) {
            spdlog::error("vkAcquireNextImageKHR failed ({})", int(acquireRes));
            frameGraph.endFrame();
            break;
        }

        ++debugFrameSerial;
        bool logFrame = (debugFrameSerial <= 10) || ((debugFrameSerial % 60) == 0);
        if (logFrame) {
            spdlog::info("Frame {} acquired image {}", debugFrameSerial, idx);
        }

        frameGraph.beginFrame();
        if (logFrame) {
            spdlog::info("Frame {} beginFrame done", debugFrameSerial);
        }

#if VOXEL_ENABLE_WINDOW
        if (glfwGetKey(window.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            frameGraph.endFrame();
            break;
        }

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

        int bState = glfwGetKey(window.handle(), GLFW_KEY_B);
        bool bDown = (bState == GLFW_PRESS);
        if (bDown && !blockPrevDown) {
            debugFlags ^= blockNormalFlag;
            spdlog::info("voxel normals {}", (debugFlags & blockNormalFlag) ? "BLOCK" : "SMOOTH");
        }
        blockPrevDown = bDown;
#endif

        render::GlobalsUBOData data{};
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

        int surfaceKeyState = glfwGetKey(window.handle(), GLFW_KEY_F);
        bool surfacePressed = (surfaceKeyState == GLFW_PRESS);
        if (surfacePressed && !surfaceTogglePrev) {
            if (cameraMode == CameraMode::FreeFly) {
                cameraMode = CameraMode::SurfaceWalk;
                auto surfaceSample = sampleSurface(camWorld);
                walkPos = surfaceSample.first;
                walkPosValid = true;
            } else {
                cameraMode = CameraMode::FreeFly;
                camWorld = walkPos;
                walkPosValid = false;
            }
        }
        surfaceTogglePrev = surfacePressed;

        glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
        if (cameraMode == CameraMode::FreeFly) {
            forward.x = std::cos(pitchRad) * std::cos(yawRad);
            forward.y = std::cos(pitchRad) * std::sin(yawRad);
            forward.z = std::sin(pitchRad);
            forward = glm::normalize(forward);
            glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
            if (glm::dot(right, right) < 1e-4f) {
                right = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            up = glm::normalize(glm::cross(right, forward));

            float moveSpeed = 5.0f;
            if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window.handle(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                moveSpeed *= 2.0f;
            }
            if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) camWorld += glm::dvec3(forward) * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) camWorld -= glm::dvec3(forward) * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) camWorld -= glm::dvec3(right)   * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) camWorld += glm::dvec3(right)   * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_SPACE) == GLFW_PRESS) camWorld += glm::dvec3(up) * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camWorld -= glm::dvec3(up) * double(moveSpeed * dt);
        } else {
            if (!walkPosValid) {
                auto surfaceSample = sampleSurface(camWorld);
                walkPos = surfaceSample.first;
                walkPosValid = true;
            }
            auto surfaceSample = sampleSurface(walkPos);
            walkPos = surfaceSample.first;
            glm::dvec3 upD = glm::normalize(surfaceSample.second);
            if (glm::length(upD) < 1e-6) {
                upD = glm::dvec3(0.0, 0.0, 1.0);
            }
            glm::dvec3 east = glm::cross(glm::dvec3(0.0, 0.0, 1.0), upD);
            if (glm::length(east) < 1e-6) {
                east = glm::dvec3(1.0, 0.0, 0.0);
            }
            east = glm::normalize(east);
            glm::dvec3 north = glm::normalize(glm::cross(upD, east));
            double cosYaw = std::cos(static_cast<double>(yawRad));
            double sinYaw = std::sin(static_cast<double>(yawRad));
            glm::dvec3 heading = glm::normalize(east * cosYaw + north * sinYaw);
            glm::dvec3 moveRight = glm::normalize(glm::cross(heading, upD));

            float moveSpeed = 5.0f;
            if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window.handle(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
                moveSpeed *= 2.0f;
            }
            if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) walkPos += heading * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) walkPos -= heading * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) walkPos -= moveRight * double(moveSpeed * dt);
            if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) walkPos += moveRight * double(moveSpeed * dt);

            surfaceSample = sampleSurface(walkPos);
            walkPos = surfaceSample.first;
            upD = glm::normalize(surfaceSample.second);
            if (glm::length(upD) < 1e-6) {
                upD = glm::dvec3(0.0, 0.0, 1.0);
            }
            glm::dvec3 cameraPosD = walkPos + upD * double(eyeHeight);
            camWorld = cameraPosD;

            double cosPitch = std::cos(static_cast<double>(pitchRad));
            double sinPitch = std::sin(static_cast<double>(pitchRad));
            glm::dvec3 viewForwardD = glm::normalize(heading * cosPitch + upD * sinPitch);
            forward = glm::vec3(viewForwardD);
            up = glm::vec3(upD);
        }
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
        glm::mat4 invV = glm::inverse(V);
        glm::mat4 invP = glm::inverse(P);
        std::memcpy(data.currView, &V[0][0], sizeof(float) * 16);
        std::memcpy(data.currProj, &P[0][0], sizeof(float) * 16);
        std::memcpy(data.currViewInv, &invV[0][0], sizeof(float) * 16);
        std::memcpy(data.currProjInv, &invP[0][0], sizeof(float) * 16);
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
        auto timings = ray.gpuTimingsMs();
        {
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
        if (statsOverlay.bricksRequestedLast > 0) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2)
               << "BUILD MS LAST " << statsOverlay.buildMsLast
               << " AVG " << statsOverlay.buildMsAvg
               << " MAX " << statsOverlay.buildMsMax;
            ss << std::setprecision(0)
               << " SAMPLES " << statsOverlay.buildSamples;
            ss << std::setprecision(0)
               << " BRICKS " << statsOverlay.bricksGeneratedLast
               << "/" << statsOverlay.bricksRequestedLast
               << " (" << std::setprecision(1) << statsOverlay.solidRatioLast * 100.0 << "%)";
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
        if (logFrame) {
            spdlog::info("Frame {} globals prepared (GPU ms: G={:.2f} T={:.2f} S={:.2f} C={:.2f})",
                         debugFrameSerial, timings[0], timings[1], timings[2], timings[3]);
        }

        if (data.frameIdx < 3) {
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

        ray.updateGlobals(vk, data);
        if (logFrame) {
            spdlog::info("Frame {} globals uploaded", debugFrameSerial);
        }

        frameGraph.addPass("PrepareSwapImage", [&, idx](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
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

        frameGraph.addPass("Raytrace", [&, idx](VkCommandBuffer cmd) {
            ray.record(vk, swap, cmd, idx);
        });

        frameGraph.addPass("OverlayHUD", [&, idx](VkCommandBuffer cmd) {
            ray.recordOverlay(cmd, idx);
        });

        frameGraph.addPass("TransitionToPresent", [&, idx](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
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

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cb, &bi);
        frameGraph.execute(cb);
        vkEndCommandBuffer(cb);
        if (logFrame) {
            spdlog::info("Frame {} command buffer recorded", debugFrameSerial);
        }

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquireSem;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &finishSem;
        auto submitStart = std::chrono::steady_clock::now();
        vkQueueSubmit(vk.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
        auto submitEnd = std::chrono::steady_clock::now();
        if (logFrame) {
            double submitMs = std::chrono::duration<double, std::milli>(submitEnd - submitStart).count();
            spdlog::info("Frame {} submitted ({:.3f} ms)", debugFrameSerial, submitMs);
        }

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        VkSwapchainKHR sc = swap.handle();
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &finishSem;
        pi.swapchainCount = 1;
        pi.pSwapchains = &sc;
        pi.pImageIndices = &idx;
        auto presentStart = std::chrono::steady_clock::now();
        vkQueuePresentKHR(vk.graphicsQueue(), &pi);
        auto presentEnd = std::chrono::steady_clock::now();
        if (logFrame) {
            double presentMs = std::chrono::duration<double, std::milli>(presentEnd - presentStart).count();
            spdlog::info("Frame {} presented ({:.3f} ms)", debugFrameSerial, presentMs);
        }
        auto waitStart = std::chrono::steady_clock::now();
        VkResult waitRes = vkQueueWaitIdle(vk.graphicsQueue());
        auto waitEnd = std::chrono::steady_clock::now();
        if (logFrame) {
            double waitMs = std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();
            spdlog::info("Frame {} queue idle ({:.3f} ms)", debugFrameSerial, waitMs);
        }

        if (waitRes != VK_SUCCESS) {
            spdlog::error("vkQueueWaitIdle failed ({})", int(waitRes));
            frameGraph.endFrame();
            break;
        }

        ray.collectGpuTimings(vk, currentFrameIdx);

        frameGraph.endFrame();
        if (logFrame) {
            spdlog::info("Frame {} endFrame done", debugFrameSerial);
        }

        // Temporarily disable per-frame debug readback to avoid blocking on device-local memory.
        // ray.readDebug(vk, currentFrameIdx);
        if (logFrame) {
            spdlog::info("Frame {} loop end", debugFrameSerial);
        }
    }

    vkDestroySemaphore(vk.device(), finishSem, nullptr); vkDestroySemaphore(vk.device(), acquireSem, nullptr);
    streaming.shutdown();
    jobs.stop();
    ray.shutdown(vk);
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
