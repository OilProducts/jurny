#pragma once

#include <cstdint>
#include <vector>
typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

struct GLFWwindow;

namespace platform {

class Window {
public:
    bool create(int width = 1280, int height = 720, const char* title = "voxel-planet");
    void poll();
    void destroy();

    // Create a Vulkan surface for this window using the given instance.
    bool createSurface(VkInstance instance, VkSurfaceKHR* outSurface) const;

    // Get instance extensions required by GLFW for surface creation.
    static void getRequiredInstanceExtensions(std::vector<const char*>& outExts);

    int width() const { return width_; }
    int height() const { return height_; }
    GLFWwindow* handle() const { return window_; }
    bool shouldClose() const;

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
};

} // namespace platform
