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

struct CameraPose {
    glm::dvec3 position = glm::dvec3(0.0);
    glm::vec3 forward = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
};

const glm::vec3 kWorldUpF(0.0f, 0.0f, 1.0f);
const glm::dvec3 kWorldUpD(0.0, 0.0, 1.0);

template <typename VecT>
VecT safeNormalizeVec(const VecT& v, const VecT& fallback) {
    using Scalar = typename VecT::value_type;
    const Scalar len = glm::length(v);
    if (!std::isfinite(static_cast<double>(len)) || len < static_cast<Scalar>(1e-6)) {
        return fallback;
    }
    return v / len;
}

template <typename VecT>
VecT safeCrossNormalizeVec(const VecT& a, const VecT& b, const VecT& fallback) {
    return safeNormalizeVec(glm::cross(a, b), fallback);
}

template <typename VecT>
VecT headingFromYaw(const VecT& east, const VecT& north, double yawRad) {
    using Scalar = typename VecT::value_type;
    const Scalar cosYaw = static_cast<Scalar>(std::cos(yawRad));
    const Scalar sinYaw = static_cast<Scalar>(std::sin(yawRad));
    return safeNormalizeVec(east * cosYaw + north * sinYaw, north);
}

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
    void handleCameraModeToggles(const InputState& input);
    CameraPose updateFreeFlyCamera(const InputState& input, float dt, float yawRad, float pitchRad);
    CameraPose updateSurfaceWalkCamera(const InputState& input, float dt, float yawRad, float pitchRad);
    CameraPose updateThirdPersonCamera(const InputState& input, float dt, float yawRad);
    void updateMeshInstances();
    void updateStreamingState(const glm::vec3& camPosF);
    void updateViewProjection(const glm::vec3& camPosF, glm::mat4& V, glm::mat4& P, glm::mat4& invV, glm::mat4& invP);
    uint32_t fillGlobalsData(const glm::mat4& V, const glm::mat4& P, const glm::mat4& invV, const glm::mat4& invP,
                             const glm::vec3& camPosF, render::GlobalsUBOData& data);
    std::vector<std::string> buildOverlayLines(float dt);
    void recordFrameGraphPasses(uint32_t imageIndex);
    bool submitFrame(FrameSync& frame, uint32_t imageIndex, VkCommandBuffer cb, bool logFrame, uint32_t currentFrameIdx);
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

    handleCameraModeToggles(input);

    const float yawRad = glm::radians(camera_.yawDeg);
    const float pitchRad = glm::radians(camera_.pitchDeg);

    CameraPose pose{camera_.position, camera_.forward, camera_.up};
    switch (camera_.mode) {
    case CameraMode::FreeFly:
        pose = updateFreeFlyCamera(input, dt, yawRad, pitchRad);
        break;
    case CameraMode::SurfaceWalk:
        pose = updateSurfaceWalkCamera(input, dt, yawRad, pitchRad);
        break;
    case CameraMode::ThirdPerson:
        pose = updateThirdPersonCamera(input, dt, yawRad);
        break;
    default:
        pose = updateFreeFlyCamera(input, dt, yawRad, pitchRad);
        break;
    }

    camera_.position = pose.position;
    camera_.forward = pose.forward;
    camera_.up = pose.up;
}

void Runtime::handleCameraModeToggles(const InputState& input) {
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
}

CameraPose Runtime::updateFreeFlyCamera(const InputState& input, float dt, float yawRad, float pitchRad) {
    CameraPose pose{};

    glm::vec3 forward{
        std::cos(pitchRad) * std::cos(yawRad),
        std::cos(pitchRad) * std::sin(yawRad),
        std::sin(pitchRad)
    };
    forward = safeNormalizeVec(forward, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 right = safeCrossNormalizeVec(forward, kWorldUpF, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 up = safeCrossNormalizeVec(right, forward, kWorldUpF);

    const float moveSpeed = input.boost ? 10.0f : 5.0f;
    glm::dvec3 move(0.0);
    if (input.moveForward) move += glm::dvec3(forward);
    if (input.moveBackward) move -= glm::dvec3(forward);
    if (input.moveLeft) move -= glm::dvec3(right);
    if (input.moveRight) move += glm::dvec3(right);
    if (input.ascend) move += glm::dvec3(up);
    if (input.descend) move -= glm::dvec3(up);
    if (glm::length(move) > 1e-6) {
        move = glm::normalize(move);
        camera_.position += move * double(moveSpeed * dt);
    }

    pose.position = camera_.position;
    pose.forward = forward;
    pose.up = up;
    return pose;
}

CameraPose Runtime::updateSurfaceWalkCamera(const InputState& input, float dt, float yawRad, float pitchRad) {
    if (!camera_.walkPosValid) {
        auto surfaceSample = sampleSurface(camera_.position);
        camera_.walkPos = surfaceSample.first;
        camera_.walkPosValid = true;
    }

    auto surfaceSample = sampleSurface(camera_.walkPos);
    camera_.walkPos = surfaceSample.first;
    glm::dvec3 upD = safeNormalizeVec(surfaceSample.second, kWorldUpD);
    glm::dvec3 east = safeCrossNormalizeVec(kWorldUpD, upD, glm::dvec3(1.0, 0.0, 0.0));
    glm::dvec3 north = safeCrossNormalizeVec(upD, east, glm::dvec3(0.0, 1.0, 0.0));
    glm::dvec3 heading = headingFromYaw(east, north, yawRad);
    glm::dvec3 moveRight = safeCrossNormalizeVec(heading, upD, glm::dvec3(1.0, 0.0, 0.0));

    const float moveSpeed = input.boost ? 10.0f : 5.0f;
    glm::dvec3 moveDelta(0.0);
    if (input.moveForward) moveDelta += heading;
    if (input.moveBackward) moveDelta -= heading;
    if (input.moveLeft) moveDelta -= moveRight;
    if (input.moveRight) moveDelta += moveRight;
    if (glm::length(moveDelta) > 1e-6) {
        moveDelta = glm::normalize(moveDelta);
        camera_.walkPos += moveDelta * double(moveSpeed * dt);
        surfaceSample = sampleSurface(camera_.walkPos);
        camera_.walkPos = surfaceSample.first;
        upD = safeNormalizeVec(surfaceSample.second, kWorldUpD);
    }

    glm::dvec3 cameraPosD = camera_.walkPos + upD * double(kEyeHeight);
    const double cosPitch = std::cos(static_cast<double>(pitchRad));
    const double sinPitch = std::sin(static_cast<double>(pitchRad));
    glm::dvec3 viewForwardD = safeNormalizeVec(heading * cosPitch + upD * sinPitch, heading);

    CameraPose pose{};
    pose.position = cameraPosD;
    pose.forward = glm::vec3(viewForwardD);
    pose.up = glm::vec3(upD);
    return pose;
}

CameraPose Runtime::updateThirdPersonCamera(const InputState& input, float dt, float yawRad) {
    if (!camera_.avatarValid) {
        auto surfaceSample = sampleSurface(camera_.position);
        camera_.avatarPos = surfaceSample.first;
        camera_.avatarValid = true;
    }

    auto surfaceSample = sampleSurface(camera_.avatarPos);
    camera_.avatarPos = surfaceSample.first;

    glm::vec3 east, north, upCheck;
    math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);
    east = safeNormalizeVec(east, glm::vec3(1.0f, 0.0f, 0.0f));
    north = safeNormalizeVec(north, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 upVec = safeNormalizeVec(glm::vec3(camera_.avatarPos), kWorldUpF);
    if (glm::length(upVec) < 1e-6f) {
        upVec = safeNormalizeVec(upCheck, kWorldUpF);
    }

    glm::vec3 heading = headingFromYaw(east, north, yawRad);
    glm::vec3 rightVec = safeCrossNormalizeVec(heading, upVec, glm::vec3(1.0f, 0.0f, 0.0f));
    heading = safeCrossNormalizeVec(upVec, rightVec, heading);

    const float moveSpeed = input.boost ? 10.0f : 5.0f;
    glm::dvec3 moveDir(0.0);
    if (input.moveForward) moveDir += glm::dvec3(heading);
    if (input.moveBackward) moveDir -= glm::dvec3(heading);
    if (input.moveLeft) moveDir -= glm::dvec3(rightVec);
    if (input.moveRight) moveDir += glm::dvec3(rightVec);
    if (glm::length(moveDir) > 1e-6) {
        moveDir = glm::normalize(moveDir);
        camera_.avatarPos += moveDir * double(moveSpeed * dt);
        auto adjustSample = sampleSurface(camera_.avatarPos);
        camera_.avatarPos = adjustSample.first;

        math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);
        east = safeNormalizeVec(east, glm::vec3(1.0f, 0.0f, 0.0f));
        north = safeNormalizeVec(north, glm::vec3(0.0f, 1.0f, 0.0f));
        upVec = safeNormalizeVec(glm::vec3(camera_.avatarPos), kWorldUpF);
        if (glm::length(upVec) < 1e-6f) {
            upVec = safeNormalizeVec(upCheck, kWorldUpF);
        }
        heading = headingFromYaw(east, north, yawRad);
        rightVec = safeCrossNormalizeVec(heading, upVec, glm::vec3(1.0f, 0.0f, 0.0f));
        heading = safeCrossNormalizeVec(upVec, rightVec, heading);
    }

    float thirdPitch = std::clamp(camera_.pitchDeg, -60.0f, 60.0f);
    camera_.pitchDeg = thirdPitch;
    const float thirdPitchRad = glm::radians(thirdPitch);
    glm::vec3 orbitDir = safeNormalizeVec(heading * std::cos(thirdPitchRad) + upVec * std::sin(thirdPitchRad), heading);
    glm::vec3 target = glm::vec3(camera_.avatarPos) + upVec * kEyeHeight;
    glm::vec3 cameraPos = target - orbitDir * kThirdPersonDistance + upVec * 0.5f;

    CameraPose pose{};
    pose.position = glm::dvec3(cameraPos);
    pose.forward = safeNormalizeVec(target - glm::vec3(pose.position), orbitDir);
    pose.up = upVec;
    return pose;
}

void Runtime::updateMeshInstances() {
    if (!meshRenderer_.ready()) {
        return;
    }

    if (camera_.mode == CameraMode::ThirdPerson && camera_.avatarValid) {
        glm::vec3 east, north, upCheck;
        math::ENU(glm::vec3(camera_.avatarPos), east, north, upCheck);

        glm::vec3 upAxis = safeNormalizeVec(glm::vec3(camera_.avatarPos), kWorldUpF);
        if (glm::length(upAxis) < 1e-6f) {
            upAxis = safeNormalizeVec(upCheck, kWorldUpF);
        }
        if (glm::length(east) < 1e-6f) {
            east = safeCrossNormalizeVec(glm::vec3(0.0f, 1.0f, 0.0f), upAxis, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        if (glm::length(north) < 1e-6f) {
            north = safeCrossNormalizeVec(upAxis, east, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        const float yawRad = glm::radians(camera_.yawDeg);
        glm::vec3 heading = headingFromYaw(east, north, yawRad);
        glm::vec3 rightAxis = safeCrossNormalizeVec(heading, upAxis, east);
        glm::vec3 forwardAxis = safeCrossNormalizeVec(upAxis, rightAxis, heading);

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

void Runtime::updateStreamingState(const glm::vec3& camPosF) {
    streaming_.update(camPosF, frameCounter_);
    const auto statsBefore = streaming_.stats();
    if ((frameCounter_ % 120u) == 0u) {
        spdlog::info("Streaming stats: selected={} queued={} building={} ready={}",
                     statsBefore.selectedRegions, statsBefore.queuedRegions, statsBefore.buildingRegions, statsBefore.readyRegions);
    }

    std::vector<world::Streaming::ReadyRegion> readyRegions;
    world::Streaming::ReadyRegion readyRegion{};
    while (streaming_.popReadyRegion(readyRegion)) {
        readyRegions.push_back(std::move(readyRegion));
    }

    if (!readyRegions.empty()) {
        size_t uploadedRegions = 0;
        size_t uploadedBricks = 0;
        for (auto& region : readyRegions) {
            const size_t bricksInRegion = region.bricks.headers.size();
            if (ray_.addRegion(vk_, region.regionCoord, std::move(region.bricks))) {
                streaming_.markRegionUploaded(region.regionCoord);
                ++uploadedRegions;
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
}

void Runtime::updateViewProjection(const glm::vec3& camPosF, glm::mat4& V, glm::mat4& P, glm::mat4& invV, glm::mat4& invP) {
    glm::vec3 forward = camera_.forward;
    glm::vec3 up = camera_.up;
    V = glm::lookAt(camPosF, camPosF + forward, up);
    float aspect = float(swap_.extent().width) / std::max(1.0f, float(swap_.extent().height));
    P = glm::perspective(glm::radians(45.0f), aspect, 0.05f, 2000.0f);
    invV = glm::inverse(V);
    invP = glm::inverse(P);

    if (meshRenderer_.ready()) {
        glm::mat4 VP = P * V;
        meshRenderer_.setCamera(VP, glm::vec3(0.4f, 1.0f, 0.3f));
    }
}

uint32_t Runtime::fillGlobalsData(const glm::mat4& V,
                                  const glm::mat4& P,
                                  const glm::mat4& invV,
                                  const glm::mat4& invP,
                                  const glm::vec3& camPosF,
                                  render::GlobalsUBOData& data) {
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
    return currentFrameIdx;
}

std::vector<std::string> Runtime::buildOverlayLines(float dt) {
    constexpr size_t kOverlayMaxCols = 128;
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

    const auto statsOverlay = streaming_.stats();
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
        double altitude = glm::length(glm::vec3(camera_.position)) - double(planetRadius_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1)
           << "CAM X " << camera_.position.x
           << " Y " << camera_.position.y
           << " Z " << camera_.position.z
           << " ALT " << altitude;
        clampLine(ss.str());
    }

    return overlayLines;
}

void Runtime::recordFrameGraphPasses(uint32_t imageIndex) {
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
}

bool Runtime::submitFrame(FrameSync& frame,
                          uint32_t imageIndex,
                          VkCommandBuffer cb,
                          bool logFrame,
                          uint32_t currentFrameIdx) {
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
        return false;
    }

    VkSwapchainKHR sc = swap_.handle();
    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
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
    return true;
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

    updateMeshInstances();

    glm::vec3 camPosF = glm::vec3(camera_.position);
    updateStreamingState(camPosF);

    glm::mat4 V{};
    glm::mat4 P{};
    glm::mat4 invV{};
    glm::mat4 invP{};
    updateViewProjection(camPosF, V, P, invV, invP);

    render::GlobalsUBOData data{};
    uint32_t currentFrameIdx = fillGlobalsData(V, P, invV, invP, camPosF, data);

    auto overlayLines = buildOverlayLines(dt);
    ray_.updateOverlayHUD(vk_, overlayLines);
    if (logFrame) {
        auto timings = ray_.gpuTimingsMs();
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

    recordFrameGraphPasses(imageIndex);

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

    if (!submitFrame(frame, imageIndex, cb, logFrame, currentFrameIdx)) {
        frameGraph_.endFrame();
        return false;
    }

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
