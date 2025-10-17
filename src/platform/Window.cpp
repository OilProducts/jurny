#include "Window.h"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <vector>

namespace platform {

static void glfwErrorCallback(int code, const char* desc) {
    spdlog::error("[glfw] error {}: {}", code, desc ? desc : "(null)");
}

bool Window::create(int w, int h, const char* title) {
    if (!glfwInit()) return false;
    glfwSetErrorCallback(glfwErrorCallback);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    window_ = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!window_) return false;
    width_ = w; height_ = h;

    // Simple key handler: ESC to close
    glfwSetKeyCallback(window_, [](GLFWwindow* win, int key, int sc, int action, int mods){
        (void)sc; (void)mods;
        if (action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }
    });
    return true;
}

void Window::poll() {
    if (window_) glfwPollEvents();
}

void Window::destroy() {
    if (window_) { glfwDestroyWindow(window_); window_ = nullptr; }
    glfwTerminate();
}

bool Window::createSurface(VkInstance instance, VkSurfaceKHR* outSurface) const {
    if (!window_) return false;
    VkSurfaceKHR surf = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, window_, nullptr, &surf) != VK_SUCCESS) return false;
    *outSurface = surf;
    return true;
}

void Window::getRequiredInstanceExtensions(std::vector<const char*>& outExts) {
    uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    if (names && count > 0) {
        for (uint32_t i=0;i<count;++i) outExts.push_back(names[i]);
    }
}

bool Window::shouldClose() const { return window_ && glfwWindowShouldClose(window_); }

} // namespace platform
