#include "App.h"
#include "Input.h"
#include "platform/VulkanContext.h"
#if VOXEL_ENABLE_WINDOW
#include "platform/Window.h"
#include "platform/Swapchain.h"
#include "render/Raytracer.h"
#include "render/MeshRenderer.h"
#endif

#include <volk.h>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <utility>
#include <sstream>
#include <iomanip>
#include <thread>
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

#if VOXEL_ENABLE_WINDOW
namespace {

constexpr float kEyeHeight = 1.7f;
constexpr float kThirdPersonDistance = 5.0f;
constexpr uint32_t kMacroSkipFlag = 8u;
constexpr uint32_t kBlockNormalFlag = 32u;

enum class CameraMode { FreeFly, SurfaceWalk, ThirdPerson };

struct CameraState {
    glm::dvec3 position = glm::dvec3(0.0);
    glm::dvec3 walkPos = glm::dvec3(0.0);
    bool walkPosValid = false;
    glm::dvec3 avatarPos = glm::dvec3(0.0);
    bool avatarValid = false;
    glm::dvec3 prevRenderOrigin = glm::dvec3(0.0);
    glm::mat4 prevView = glm::mat4(1.0f);
    glm::mat4 prevProj = glm::mat4(1.0f);
    float yawDeg = 180.0f;
    float pitchDeg = 0.0f;
    CameraMode mode = CameraMode::FreeFly;
    glm::vec3 forward = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
};

struct FrameSync {
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
};

class Runtime {
public:
    int run();

private:
    bool initialize();
    bool initWindowAndDevice();
    bool initAssets();
    bool initRenderers();
    bool initStreaming();
    bool initFrameResources();
    void initCamera();
    void mainLoop();
    void shutdown();

    void pumpEvents();
    bool waitOnFence(VkFence fence);
    bool waitForFrame(FrameSync& frame);
    bool acquireSwapImage(FrameSync& frame, uint32_t& imageIndex);
    void updateDebugFlags(const InputState& input);
    void updateCamera(const InputState& input, float dt);
    bool buildFrame(FrameSync& frame, uint32_t imageIndex, float dt);
    std::pair<glm::dvec3, glm::dvec3> sampleSurface(const glm::dvec3& guess) const;

private:
    platform::VulkanContext vk_;
    platform::Window window_;
    platform::Swapchain swap_;
    core::Jobs jobs_;
    world::Streaming streaming_;
    core::FrameGraph frameGraph_;
    render::Raytracer ray_;
    render::MeshRenderer meshRenderer_;
    core::AssetRegistry assetRegistry_;
    Input input_;

    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<FrameSync> frames_;
    std::vector<VkFence> imagesInFlight_;

    uint32_t maxFramesInFlight_ = 0u;
    uint32_t currentFrame_ = 0u;
    bool assetsReady_ = false;
    bool enableStreaming_ = true;
    const world::BrickStore* brickStore_ = nullptr;
    float planetRadius_ = 100.0f;
    uint32_t debugFlags_ = 0u;
    uint32_t frameCounter_ = 0u;
    size_t debugFrameSerial_ = 0u;
    CameraState camera_;
    std::chrono::steady_clock::time_point lastTime_{};
};

} // namespace
#endif

int App::run() {
#if VOXEL_ENABLE_WINDOW
    Runtime runtime;
    return runtime.run();
#else
    platform::VulkanContext vk;
    bool enableValidation = true;
    if (!vk.initInstance({}, enableValidation)) return 1;
    if (!vk.initDevice(VK_NULL_HANDLE)) return 1;
    vk.shutdown();
    return 0;
#endif
}

#if VOXEL_ENABLE_WINDOW

int Runtime::run() {
    if (!initialize()) {
        shutdown();
        return 1;
    }
    mainLoop();
    shutdown();
    return 0;
}

bool Runtime::initialize() {
    if (!initWindowAndDevice()) {
        return false;
    }
    jobs_.start();
    initAssets();
    if (!initRenderers()) {
        return false;
    }
    if (!initStreaming()) {
        return false;
    }
    if (!initFrameResources()) {
        return false;
    }
    input_.initialize(window_);
    initCamera();
    lastTime_ = std::chrono::steady_clock::now();
    return true;
}

bool Runtime::initWindowAndDevice() {
    bool enableValidation = true;
    spdlog::info("Creating window");
    if (!window_.create()) {
        spdlog::error("Window creation failed");
        return false;
    }
    spdlog::info("Window created ({}x{})", window_.width(), window_.height());

    std::vector<const char*> instanceExts;
    platform::Window::getRequiredInstanceExtensions(instanceExts);
    spdlog::info("Initializing Vulkan instance");
    if (!vk_.initInstance(instanceExts, enableValidation)) {
        spdlog::error("initInstance failed");
        return false;
    }
    spdlog::info("Vulkan instance ready");

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!window_.createSurface(vk_.instance(), &surface)) {
        spdlog::error("createSurface failed");
        return false;
    }
    spdlog::info("Surface created");

    if (!vk_.initDevice(surface)) {
        spdlog::error("initDevice failed");
        return false;
    }
    spdlog::info("Device initialized");

    platform::SwapchainCreateInfo sci{};
    sci.device = vk_.device();
    sci.physicalDevice = vk_.physicalDevice();
    sci.surface = surface;
    sci.graphicsQueueFamily = vk_.graphicsFamily();
    sci.presentQueueFamily = vk_.graphicsFamily();
    sci.width = window_.width();
    sci.height = window_.height();
    spdlog::info("Creating swapchain");
    if (!swap_.create(sci)) {
        spdlog::error("Swapchain creation failed");
        return false;
    }
    spdlog::info("Swapchain ready");
    return true;
}

bool Runtime::initAssets() {
    const char* envAssets = std::getenv("VOXEL_ASSETS_DIR");
    const char* assetsDir = envAssets ? envAssets :
#ifdef VOXEL_ASSETS_DIR
        VOXEL_ASSETS_DIR;
#else
        "assets";
#endif
    std::string assetRoot = assetsDir ? assetsDir : std::string{};
    if (assetRoot.empty()) {
        spdlog::warn("Asset directory not specified; proceeding without packed assets");
        return true;
    }

    if (assetRegistry_.initialize(assetRoot)) {
        assetsReady_ = true;
        spdlog::info("Asset pack loaded: {} entries", assetRegistry_.assetList().size());
        if (!assetRegistry_.contains("materials.json")) {
            spdlog::warn("materials.json not found in asset pack");
        }
    } else {
        spdlog::warn("Failed to initialize asset registry at '{}'", assetRoot);
    }
    return true;
}

bool Runtime::initRenderers() {
    if (assetsReady_) {
        ray_.setAssetRegistry(&assetRegistry_);
    }
    if (!ray_.init(vk_, swap_)) {
        spdlog::error("Raytracer init failed.");
        return false;
    }
    if (!meshRenderer_.init(vk_, swap_)) {
        spdlog::warn("Mesh renderer initialization failed; avatar rendering disabled");
    }
    brickStore_ = ray_.worldStore();
    planetRadius_ = brickStore_ ? static_cast<float>(brickStore_->params().R) : 100.0f;
    spdlog::debug("Raytracer initialized");
    return true;
}

bool Runtime::initStreaming() {
    if (!enableStreaming_) {
        spdlog::info("Streaming disabled for debugging");
        return true;
    }
    if (!brickStore_) {
        spdlog::warn("Streaming not initialized: no brick store available");
        return true;
    }

    const auto& planetParams = brickStore_->params();
    const float innerBand = static_cast<float>(planetParams.T);
    const float heightSpan = static_cast<float>(planetParams.Hmax);
    const float shellMargin = brickStore_->brickSize();
    world::Streaming::Config streamCfg;
    streamCfg.shellInner = planetRadius_ - std::max(innerBand, 6.0f);
    streamCfg.shellOuter = planetRadius_ + heightSpan + shellMargin;
    streamCfg.keepRadius = 140.0f;
    streamCfg.loadRadius = 180.0f;
    streamCfg.simRadius = 140.0f;
    streamCfg.regionDimBricks = 8;
    streamCfg.maxRegionSelectionsPerFrame = 24;
    streamCfg.maxConcurrentGenerations = std::max(8, static_cast<int>(jobs_.workerCount()));
    streaming_.initialize(*brickStore_, streamCfg, &jobs_);
    spdlog::debug("Streaming initialized");
    return true;
}

bool Runtime::initFrameResources() {
    maxFramesInFlight_ = std::min<uint32_t>(swap_.imageCount(), 2u);
    if (maxFramesInFlight_ == 0u) {
        maxFramesInFlight_ = 1u;
    }

    commandBuffers_.resize(maxFramesInFlight_);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk_.commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = maxFramesInFlight_;
    if (vkAllocateCommandBuffers(vk_.device(), &cbai, commandBuffers_.data()) != VK_SUCCESS) {
        spdlog::error("Failed to allocate command buffers");
        return false;
    }

    frames_.assign(maxFramesInFlight_, FrameSync{});
    imagesInFlight_.assign(swap_.imageCount(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto& frame : frames_) {
        if (vkCreateSemaphore(vk_.device(), &sci, nullptr, &frame.imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(vk_.device(), &sci, nullptr, &frame.renderFinished) != VK_SUCCESS ||
            vkCreateFence(vk_.device(), &fci, nullptr, &frame.inFlight) != VK_SUCCESS) {
            spdlog::error("Failed to create per-frame synchronization primitives");
            return false;
        }
    }
    currentFrame_ = 0;
    return true;
}

void Runtime::initCamera() {
    camera_.position = glm::dvec3(double(planetRadius_) + 40.0, 0.0, 0.0);
    camera_.walkPos = camera_.position;
    camera_.avatarPos = camera_.position;
    camera_.walkPosValid = false;
    camera_.avatarValid = false;
    camera_.prevRenderOrigin = camera_.position;
    camera_.prevView = glm::mat4(1.0f);
    camera_.prevProj = glm::mat4(1.0f);
    camera_.yawDeg = 180.0f;
    camera_.pitchDeg = 0.0f;
    camera_.mode = CameraMode::FreeFly;
    camera_.forward = glm::vec3(0.0f, 0.0f, 1.0f);
    camera_.up = glm::vec3(0.0f, 0.0f, 1.0f);
}

void Runtime::mainLoop() {
    debugFrameSerial_ = 0;
    while (!window_.shouldClose()) {
        FrameSync& frame = frames_[currentFrame_];
        if (!waitForFrame(frame)) {
            break;
        }
        window_.poll();

        uint32_t imageIndex = 0;
        if (!acquireSwapImage(frame, imageIndex)) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime_).count();
        lastTime_ = now;

        InputState inputState = input_.sample(window_);
        if (inputState.requestExit) {
            break;
        }

        updateDebugFlags(inputState);
        updateCamera(inputState, dt);
        if (!buildFrame(frame, imageIndex, dt)) {
            break;
        }

        currentFrame_ = (currentFrame_ + 1) % maxFramesInFlight_;
    }
}

void Runtime::shutdown() {
    if (!commandBuffers_.empty() && vk_.device()) {
        vkFreeCommandBuffers(vk_.device(), vk_.commandPool(), static_cast<uint32_t>(commandBuffers_.size()), commandBuffers_.data());
    }
    commandBuffers_.clear();

    if (vk_.device()) {
        for (auto& frame : frames_) {
            if (frame.imageAvailable) vkDestroySemaphore(vk_.device(), frame.imageAvailable, nullptr);
            if (frame.renderFinished) vkDestroySemaphore(vk_.device(), frame.renderFinished, nullptr);
            if (frame.inFlight) vkDestroyFence(vk_.device(), frame.inFlight, nullptr);
        }
    }
    frames_.clear();
    imagesInFlight_.clear();

    streaming_.shutdown();
    jobs_.stop();
    meshRenderer_.shutdown(vk_);
    ray_.shutdown(vk_);
    if (vk_.device()) {
        swap_.destroy(vk_.device());
    }
    vk_.shutdown();
    window_.destroy();
}

void Runtime::pumpEvents() {
    window_.poll();
    glfwWaitEventsTimeout(0.001);
}

bool Runtime::waitOnFence(VkFence fence) {
    if (fence == VK_NULL_HANDLE) {
        return true;
    }
    while (true) {
        VkResult res = vkWaitForFences(vk_.device(), 1, &fence, VK_TRUE, 0);
        if (res == VK_SUCCESS) {
            return true;
        }
        if (res == VK_TIMEOUT) {
            pumpEvents();
            continue;
        }
        spdlog::error("vkWaitForFences failed ({})", int(res));
        return false;
    }
}

bool Runtime::waitForFrame(FrameSync& frame) {
    return waitOnFence(frame.inFlight);
}

bool Runtime::acquireSwapImage(FrameSync& frame, uint32_t& imageIndex) {
    VkResult acquireRes = vkAcquireNextImageKHR(vk_.device(), swap_.handle(), UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquireRes != VK_SUCCESS) {
        spdlog::error("vkAcquireNextImageKHR failed ({})", int(acquireRes));
        return false;
    }

    if (imageIndex < imagesInFlight_.size() && imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
        if (!waitOnFence(imagesInFlight_[imageIndex])) {
            spdlog::error("Failed to wait on image fence");
            return false;
        }
    }
    if (imageIndex < imagesInFlight_.size()) {
        imagesInFlight_[imageIndex] = frame.inFlight;
    }

    if (vkResetFences(vk_.device(), 1, &frame.inFlight) != VK_SUCCESS) {
        spdlog::error("vkResetFences failed");
        return false;
    }

    ++debugFrameSerial_;
    return true;
}

void Runtime::updateDebugFlags(const InputState& input) {
    if (input.debugPreset.has_value()) {
        debugFlags_ = *input.debugPreset;
    }
    if (input.toggleMacroSkip) {
        debugFlags_ ^= kMacroSkipFlag;
        spdlog::info("macro-skip {}", (debugFlags_ & kMacroSkipFlag) ? "ON" : "OFF");
    }
    if (input.toggleBlockNormals) {
        debugFlags_ ^= kBlockNormalFlag;
        spdlog::info("voxel normals {}", (debugFlags_ & kBlockNormalFlag) ? "BLOCK" : "SMOOTH");
    }
}

void Runtime::updateCamera(const InputState& input, float dt) {
    camera_.yawDeg += input.lookDelta.x;
    camera_.pitchDeg += input.lookDelta.y;
    camera_.pitchDeg = std::clamp(camera_.pitchDeg, -89.0f, 89.0f);

    if (input.toggleSurfaceWalk) {
        if (camera_.mode == CameraMode::FreeFly) {
            camera_.mode = CameraMode::SurfaceWalk;
            auto surfaceSample = sampleSurface(camera_.position);
            camera_.walkPos = surfaceSample.first;
            camera_.walkPosValid = true;
        } else {
            camera_.mode = CameraMode::FreeFly;
            camera_.position = camera_.walkPos;
            camera_.walkPosValid = false;
        }
    }

    if (input.toggleThirdPerson) {
        if (camera_.mode == CameraMode::ThirdPerson) {
            camera_.mode = CameraMode::FreeFly;
            camera_.avatarValid = false;
        } else {
            camera_.mode = CameraMode::ThirdPerson;
            auto surfaceSample = sampleSurface(camera_.position);
            camera_.avatarPos = surfaceSample.first;
            camera_.avatarValid = true;
        }
    }

    const float yawRad = glm::radians(camera_.yawDeg);
    const float pitchRad = glm::radians(camera_.pitchDeg);
    glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
    glm::vec3 forward;
    glm::vec3 up(0.0f, 0.0f, 1.0f);

    if (camera_.mode == CameraMode::FreeFly) {
        forward.x = std::cos(pitchRad) * std::cos(yawRad);
        forward.y = std::cos(pitchRad) * std::sin(yawRad);
        forward.z = std::sin(pitchRad);
        forward = glm::normalize(forward);
        glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        if (glm::dot(right, right) < 1e-4f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        up = glm::normalize(glm::cross(right, forward));

        float moveSpeed = input.boost ? 10.0f : 5.0f;
        if (input.moveForward) camera_.position += glm::dvec3(forward) * double(moveSpeed * dt);
        if (input.moveBackward) camera_.position -= glm::dvec3(forward) * double(moveSpeed * dt);
        if (input.moveLeft) camera_.position -= glm::dvec3(right) * double(moveSpeed * dt);
        if (input.moveRight) camera_.position += glm::dvec3(right) * double(moveSpeed * dt);
        if (input.ascend) camera_.position += glm::dvec3(up) * double(moveSpeed * dt);
        if (input.descend) camera_.position -= glm::dvec3(up) * double(moveSpeed * dt);
    } else if (camera_.mode == CameraMode::SurfaceWalk) {
        if (!camera_.walkPosValid) {
            auto surfaceSample = sampleSurface(camera_.position);
            camera_.walkPos = surfaceSample.first;
            camera_.walkPosValid = true;
        }
        auto surfaceSample = sampleSurface(camera_.walkPos);
        camera_.walkPos = surfaceSample.first;
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

        float moveSpeed = input.boost ? 10.0f : 5.0f;
        if (input.moveForward) camera_.walkPos += heading * double(moveSpeed * dt);
        if (input.moveBackward) camera_.walkPos -= heading * double(moveSpeed * dt);
        if (input.moveLeft) camera_.walkPos -= moveRight * double(moveSpeed * dt);
        if (input.moveRight) camera_.walkPos += moveRight * double(moveSpeed * dt);

        surfaceSample = sampleSurface(camera_.walkPos);
        camera_.walkPos = surfaceSample.first;
        upD = glm::normalize(surfaceSample.second);
        if (glm::length(upD) < 1e-6) {
            upD = glm::dvec3(0.0, 0.0, 1.0);
        }
        glm::dvec3 cameraPosD = camera_.walkPos + upD * double(kEyeHeight);
        camera_.position = cameraPosD;

        double cosPitch = std::cos(static_cast<double>(pitchRad));
        double sinPitch = std::sin(static_cast<double>(pitchRad));
        glm::dvec3 viewForwardD = glm::normalize(heading * cosPitch + upD * sinPitch);
        forward = glm::vec3(viewForwardD);
        up = glm::vec3(upD);
    } else if (camera_.mode == CameraMode::ThirdPerson) {
        if (!camera_.avatarValid) {
            auto surfaceSample = sampleSurface(camera_.position);
            camera_.avatarPos = surfaceSample.first;
            camera_.avatarValid = true;
        }

        auto surfaceSample = sampleSurface(camera_.avatarPos);
        camera_.avatarPos = surfaceSample.first;
        glm::vec3 upVec = glm::normalize(glm::vec3(camera_.avatarPos));
        if (!std::isfinite(upVec.x) || glm::length(upVec) < 1e-6f) {
            upVec = worldUp;
        }

        glm::vec3 east, north, upCheck;
        math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);
        if (glm::length(upVec) < 1e-6f) upVec = upCheck;
        double cosYaw = std::cos(static_cast<double>(yawRad));
        double sinYaw = std::sin(static_cast<double>(yawRad));
        glm::vec3 heading = glm::normalize(east * static_cast<float>(cosYaw) + north * static_cast<float>(sinYaw));
        glm::vec3 rightVec = glm::normalize(glm::cross(heading, upVec));
        if (!std::isfinite(rightVec.x) || glm::length(rightVec) < 1e-6f) {
            rightVec = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        heading = glm::normalize(glm::cross(upVec, rightVec));

        glm::dvec3 moveDir(0.0);
        float moveSpeed = input.boost ? 10.0f : 5.0f;
        if (input.moveForward) moveDir += glm::dvec3(heading);
        if (input.moveBackward) moveDir -= glm::dvec3(heading);
        if (input.moveLeft) moveDir -= glm::dvec3(rightVec);
        if (input.moveRight) moveDir += glm::dvec3(rightVec);
        if (glm::length(moveDir) > 1e-6) {
            moveDir = glm::normalize(moveDir);
            camera_.avatarPos += moveDir * double(moveSpeed * dt);
            auto adjustSample = sampleSurface(camera_.avatarPos);
            camera_.avatarPos = adjustSample.first;
            upVec = glm::normalize(glm::vec3(camera_.avatarPos));
            if (!std::isfinite(upVec.x) || glm::length(upVec) < 1e-6f) {
                upVec = worldUp;
            }
            math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);
            if (glm::length(upVec) < 1e-6f) upVec = upCheck;
            heading = glm::normalize(east * static_cast<float>(cosYaw) + north * static_cast<float>(sinYaw));
            rightVec = glm::normalize(glm::cross(heading, upVec));
            if (!std::isfinite(rightVec.x) || glm::length(rightVec) < 1e-6f) {
                rightVec = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            heading = glm::normalize(glm::cross(upVec, rightVec));
        }

        float thirdPitch = std::clamp(camera_.pitchDeg, -60.0f, 60.0f);
        camera_.pitchDeg = thirdPitch;
        float thirdPitchRad = glm::radians(thirdPitch);
        glm::vec3 orbitDir = glm::normalize(heading * std::cos(thirdPitchRad) + upVec * std::sin(thirdPitchRad));
        glm::vec3 target = glm::vec3(camera_.avatarPos) + upVec * kEyeHeight;
        glm::vec3 cameraPos = target - orbitDir * kThirdPersonDistance + upVec * 0.5f;
        camera_.position = glm::dvec3(cameraPos);
        forward = glm::normalize(target - glm::vec3(camera_.position));
        up = upVec;
    }

    camera_.forward = forward;
    camera_.up = up;
}

bool Runtime::buildFrame(FrameSync& frame, uint32_t imageIndex, float dt) {
    bool logFrame = (debugFrameSerial_ <= 10) || ((debugFrameSerial_ % 60) == 0);
    if (logFrame) {
        spdlog::info("Frame {} acquired image {}", debugFrameSerial_, imageIndex);
    }

    frameGraph_.beginFrame();
    if (logFrame) {
        spdlog::info("Frame {} beginFrame done", debugFrameSerial_);
    }

    glm::vec3 forward = camera_.forward;
    glm::vec3 up = camera_.up;

    if (meshRenderer_.ready()) {
        if (camera_.mode == CameraMode::ThirdPerson && camera_.avatarValid) {
            glm::vec3 east, north, upCheck;
            math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);

            glm::vec3 upAxis = glm::normalize(glm::vec3(camera_.avatarPos));
            if (!std::isfinite(upAxis.x) || glm::length(upAxis) < 1e-6f) {
                upAxis = upCheck;
            }
            if (!std::isfinite(upAxis.x) || glm::length(upAxis) < 1e-6f) {
                upAxis = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            if (glm::length(east) < 1e-6f) {
                east = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), upAxis));
            }
            if (glm::length(east) < 1e-6f) {
                east = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            if (glm::length(north) < 1e-6f) {
                north = glm::normalize(glm::cross(upAxis, east));
            }
            double cosYaw = std::cos(glm::radians(camera_.yawDeg));
            double sinYaw = std::sin(glm::radians(camera_.yawDeg));
            glm::vec3 heading = glm::normalize(east * static_cast<float>(cosYaw) + north * static_cast<float>(sinYaw));
            if (glm::length(heading) < 1e-6f) heading = north;
            glm::vec3 rightAxis = glm::normalize(glm::cross(heading, upAxis));
            if (!std::isfinite(rightAxis.x) || glm::length(rightAxis) < 1e-6f) {
                rightAxis = east;
            }
            glm::vec3 forwardAxis = glm::normalize(glm::cross(upAxis, rightAxis));

            glm::mat4 model(1.0f);
            model[0] = glm::vec4(rightAxis, 0.0f);
            model[1] = glm::vec4(upAxis, 0.0f);
            model[2] = glm::vec4(forwardAxis, 0.0f);
            model[3] = glm::vec4(glm::vec3(camera_.avatarPos), 1.0f);
            render::MeshInstance inst{};
            inst.model = model;
            inst.color = glm::vec4(0.85f, 0.2f, 0.95f, 1.0f);
            meshRenderer_.updateInstances(std::vector<render::MeshInstance>{inst});
        } else {
            meshRenderer_.updateInstances(std::vector<render::MeshInstance>{});
        }
    }

    glm::vec3 camPosF = glm::vec3(camera_.position);
    streaming_.update(camPosF, frameCounter_);
    const auto statsBefore = streaming_.stats();
    if ((frameCounter_ % 120u) == 0u) {
        spdlog::info("Streaming stats: selected={} queued={} building={} ready={}",
                     statsBefore.selectedRegions, statsBefore.queuedRegions, statsBefore.buildingRegions, statsBefore.readyRegions);
    }

    std::vector<world::Streaming::ReadyRegion> readyRegionsFrame;
    world::Streaming::ReadyRegion readyRegion{};
    while (streaming_.popReadyRegion(readyRegion)) {
        readyRegionsFrame.push_back(std::move(readyRegion));
    }

    if (!readyRegionsFrame.empty()) {
        size_t uploadedRegions = 0;
        size_t uploadedBricks = 0;
        for (auto& region : readyRegionsFrame) {
            const size_t bricksInRegion = region.bricks.headers.size();
            if (ray_.addRegion(vk_, region.regionCoord, std::move(region.bricks))) {
                streaming_.markRegionUploaded(region.regionCoord);
                uploadedRegions++;
                uploadedBricks += bricksInRegion;
            }
        }
        if (uploadedRegions > 0) {
            spdlog::info("Streaming commit applied: regions={} bricks={} (total resident={})",
                         uploadedRegions,
                         uploadedBricks,
                         ray_.brickCount());
        }
    }

    glm::ivec3 evictCoord;
    while (streaming_.popEvictedRegion(evictCoord)) {
        if (ray_.removeRegion(vk_, evictCoord)) {
            streaming_.markRegionEvicted(evictCoord);
        }
    }

    glm::mat4 V = glm::lookAt(camPosF, camPosF + forward, up);
    float aspect = float(swap_.extent().width) / float(swap_.extent().height);
    glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.05f, 2000.0f);
    glm::mat4 invV = glm::inverse(V);
    glm::mat4 invP = glm::inverse(P);
    if (meshRenderer_.ready()) {
        glm::mat4 VP = P * V;
        meshRenderer_.setCamera(VP, glm::vec3(0.4f, 1.0f, 0.3f));
    }

    render::GlobalsUBOData data{};
    std::memcpy(data.currView, &V[0][0], sizeof(float) * 16);
    std::memcpy(data.currProj, &P[0][0], sizeof(float) * 16);
    std::memcpy(data.currViewInv, &invV[0][0], sizeof(float) * 16);
    std::memcpy(data.currProjInv, &invP[0][0], sizeof(float) * 16);
    std::memcpy(data.prevView, &camera_.prevView[0][0], sizeof(float) * 16);
    std::memcpy(data.prevProj, &camera_.prevProj[0][0], sizeof(float) * 16);
    data.renderOrigin[0] = camPosF.x;
    data.renderOrigin[1] = camPosF.y;
    data.renderOrigin[2] = camPosF.z;
    data.renderOrigin[3] = 0.0f;
    glm::vec3 originDelta = glm::vec3(camera_.prevRenderOrigin - camera_.position);
    data.originDeltaPrevToCurr[0] = originDelta.x;
    data.originDeltaPrevToCurr[1] = originDelta.y;
    data.originDeltaPrevToCurr[2] = originDelta.z;
    data.originDeltaPrevToCurr[3] = 0.0f;

    float voxelSize = 0.5f;
    float brickSize = 4.0f;
    float Rin = planetRadius_ - 20.0f;
    float Rout = planetRadius_ + 80.0f;
    float Rsea = planetRadius_;
    float planetRadiusForUBO = planetRadius_;
    float warpBand = 8.0f * brickSize;
    if (brickStore_) {
        voxelSize = brickStore_->voxelSize();
        brickSize = brickStore_->brickSize();
        warpBand = 8.0f * brickSize;
        const auto& params = brickStore_->params();
        const float trenchDepth = static_cast<float>(params.T);
        const float maxHeight = static_cast<float>(params.Hmax);
        const float seaRadius = static_cast<float>(params.sea);
        const float innerBand = std::max(trenchDepth, 6.0f);
        const float shellMargin = brickSize;
        Rin = std::max(planetRadius_ - innerBand, 0.0f);
        Rout = planetRadius_ + maxHeight + shellMargin;
        if (Rout <= Rin) {
            Rout = Rin + std::max(shellMargin, 1.0f);
        }
        const float hasSea = (seaRadius > 0.0f) ? seaRadius : planetRadius_;
        if (seaRadius > 0.0f && seaRadius != planetRadius_) {
            Rsea = hasSea;
            Rin = std::max(0.0f, std::min(Rin, Rsea - warpBand));
            Rout = std::max(Rout, Rsea + warpBand);
        } else {
            Rsea = planetRadius_;
        }
        planetRadiusForUBO = static_cast<float>(params.R);
    }
    data.voxelSize = voxelSize;
    data.brickSize = brickSize;
    data.Rin = Rin;
    data.Rout = Rout;
    data.Rsea = Rsea;
    data.planetRadius = planetRadiusForUBO;
    data.exposure = 1.0f;
    data.frameIdx = frameCounter_;
    data.maxBounces = 1;
    data.width = swap_.extent().width;
    data.height = swap_.extent().height;
    data.raysPerPixel = 1;
    data.flags = debugFlags_;
    uint32_t currentFrameIdx = data.frameIdx;
    ++frameCounter_;
    camera_.prevView = V;
    camera_.prevProj = P;
    camera_.prevRenderOrigin = camera_.position;

    constexpr size_t kOverlayMaxCols = 96;
    const auto statsOverlay = streaming_.stats();
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
    auto timings = ray_.gpuTimingsMs();
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
        ss << "BRICKS " << ray_.brickCount()
           << " REG " << ray_.residentRegionCount()
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
        double altitude = glm::length(camPosF) - double(planetRadius_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1)
           << "CAM X " << camPosF.x
           << " Y " << camPosF.y
           << " Z " << camPosF.z
           << " ALT " << altitude;
        clampLine(ss.str());
    }
    ray_.updateOverlayHUD(vk_, overlayLines);
    if (logFrame) {
        spdlog::info("Frame {} globals prepared (GPU ms: G={:.2f} T={:.2f} S={:.2f} C={:.2f})",
                     debugFrameSerial_, timings[0], timings[1], timings[2], timings[3]);
    }

    if (data.frameIdx < 3) {
        auto makeRayCPU = [&](int px, int py) {
            float w = float(swap_.extent().width), h = float(swap_.extent().height);
            glm::vec2 uv = (glm::vec2(px + 0.5f, py + 0.5f)) / glm::vec2(w, h);
            glm::vec2 ndc = glm::vec2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
            glm::vec4 pView = invP * glm::vec4(ndc, 1.0f, 1.0f);
            pView /= std::max(pView.w, 1e-6f);
            glm::vec3 ro = glm::vec3(invV * glm::vec4(0, 0, 0, 1));
            glm::vec3 dirWorld = glm::vec3(invV * glm::vec4(glm::normalize(glm::vec3(pView)), 0));
            glm::vec3 rd = glm::normalize(dirWorld);
            return std::pair<glm::vec3, glm::vec3>(ro, rd);
        };
        auto [ro0, rd0] = makeRayCPU(swap_.extent().width / 2, swap_.extent().height / 2);
        auto [ro1, rd1] = makeRayCPU(0, 0);
        float tEnter = 0, tExit = 0;
        bool hit = math::IntersectSphereShell(ro0, rd0, data.Rin, data.Rout, tEnter, tExit);
        spdlog::info("Extent={}x{} Rin={} Rout={} eye=({:.1f},{:.1f},{:.1f})",
                     swap_.extent().width, swap_.extent().height, data.Rin, data.Rout, camPosF.x, camPosF.y, camPosF.z);
        spdlog::info("Center ray ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f}) hit={} t=[{:.1f},{:.1f}]",
                     ro0.x, ro0.y, ro0.z, rd0.x, rd0.y, rd0.z, hit ? 1 : 0, tEnter, tExit);
        spdlog::info("Corner ray  ro=({:.1f},{:.1f},{:.1f}) rd=({:.3f},{:.3f},{:.3f})",
                     ro1.x, ro1.y, ro1.z, rd1.x, rd1.y, rd1.z);
        float dotDirs = glm::dot(rd0, rd1);
        spdlog::info("dot(center,corner)={:.3f} (expect < 1)", dotDirs);
    }

    ray_.updateGlobals(vk_, data);
    if (logFrame) {
        spdlog::info("Frame {} globals uploaded", debugFrameSerial_);
    }

    frameGraph_.addPass("PrepareSwapImage", [&, imageIndex](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = swap_.image(imageIndex);
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

    frameGraph_.addPass("Raytrace", [&, imageIndex](VkCommandBuffer cmd) {
        ray_.record(vk_, swap_, cmd, imageIndex);
    });

    frameGraph_.addPass("MeshAvatar", [&, imageIndex](VkCommandBuffer cmd) {
        if (!meshRenderer_.ready() || !meshRenderer_.hasInstances()) {
            return;
        }
        VkImage image = swap_.image(imageIndex);
        VkImageView view = swap_.imageViews()[imageIndex];
        meshRenderer_.record(cmd, image, view);
    });

    frameGraph_.addPass("OverlayHUD", [&, imageIndex](VkCommandBuffer cmd) {
        ray_.recordOverlay(cmd, imageIndex);
    });

    frameGraph_.addPass("TransitionToPresent", [&, imageIndex](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = swap_.image(imageIndex);
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);
    });

    VkCommandBuffer cb = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);
    frameGraph_.execute(cb);
    vkEndCommandBuffer(cb);
    if (logFrame) {
        spdlog::info("Frame {} command buffer recorded", debugFrameSerial_);
    }

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &frame.imageAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &frame.renderFinished;
    auto submitStart = std::chrono::steady_clock::now();
    VkResult submitRes = vkQueueSubmit(vk_.graphicsQueue(), 1, &si, frame.inFlight);
    auto submitEnd = std::chrono::steady_clock::now();
    if (logFrame) {
        double submitMs = std::chrono::duration<double, std::milli>(submitEnd - submitStart).count();
        spdlog::info("Frame {} submitted ({:.3f} ms)", debugFrameSerial_, submitMs);
    }
    if (submitRes != VK_SUCCESS) {
        spdlog::error("vkQueueSubmit failed ({})", int(submitRes));
        frameGraph_.endFrame();
        return false;
    }

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    VkSwapchainKHR sc = swap_.handle();
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &frame.renderFinished;
    pi.swapchainCount = 1;
    pi.pSwapchains = &sc;
    pi.pImageIndices = &imageIndex;
    auto presentStart = std::chrono::steady_clock::now();
    vkQueuePresentKHR(vk_.graphicsQueue(), &pi);
    auto presentEnd = std::chrono::steady_clock::now();
    if (logFrame) {
        double presentMs = std::chrono::duration<double, std::milli>(presentEnd - presentStart).count();
        spdlog::info("Frame {} presented ({:.3f} ms)", debugFrameSerial_, presentMs);
    }

    ray_.collectGpuTimings(vk_, currentFrameIdx);

    frameGraph_.endFrame();
    if (logFrame) {
        spdlog::info("Frame {} endFrame done", debugFrameSerial_);
    }
    return true;
}

std::pair<glm::dvec3, glm::dvec3> Runtime::sampleSurface(const glm::dvec3& guess) const {
    if (!brickStore_) {
        glm::dvec3 dir = glm::length(guess) > 1e-6 ? glm::normalize(guess) : glm::dvec3(0.0, 0.0, 1.0);
        glm::dvec3 pos = dir * double(planetRadius_);
        return {pos, dir};
    }
    glm::vec3 p = glm::vec3(guess);
    const float eps = brickStore_->voxelSize();
    glm::vec3 normal = glm::normalize(brickStore_->worldGen().crustNormal(p, eps));
    for (int i = 0; i < 8; ++i) {
        float field = brickStore_->worldGen().crustField(p);
        if (!std::isfinite(field)) break;
        if (std::abs(field) < 0.01f) break;
        normal = glm::normalize(brickStore_->worldGen().crustNormal(p, eps));
        p -= normal * field;
    }
    glm::dvec3 surfacePos = glm::dvec3(p);
    glm::dvec3 radial = glm::length(surfacePos) > 1e-6
        ? glm::normalize(surfacePos)
        : glm::dvec3(0.0, 0.0, 1.0);
    return {surfacePos, radial};
}

#endif // VOXEL_ENABLE_WINDOW

} // namespace app
