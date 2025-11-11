#include "Input.h"
#include "platform/Window.h"

#if VOXEL_ENABLE_WINDOW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace app {

void Input::initialize(platform::Window& window) {
#if VOXEL_ENABLE_WINDOW
    GLFWwindow* handle = window.handle();
    if (!handle) return;
    glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    rawMouseSupported_ = glfwRawMouseMotionSupported() == GLFW_TRUE;
    if (rawMouseSupported_) {
        glfwSetInputMode(handle, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    }
#else
    (void)window;
#endif
}

InputState Input::sample(platform::Window& window) {
    InputState state{};
#if VOXEL_ENABLE_WINDOW
    GLFWwindow* handle = window.handle();
    if (!handle) {
        return state;
    }

    int captureState = glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT);
    bool capturePressed = (captureState == GLFW_PRESS);
    if (capturePressed != cursorCaptured_) {
        cursorCaptured_ = capturePressed;
        int cursorMode = cursorCaptured_ ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL;
        glfwSetInputMode(handle, GLFW_CURSOR, cursorMode);
        if (rawMouseSupported_) {
            glfwSetInputMode(handle, GLFW_RAW_MOUSE_MOTION, cursorCaptured_ ? GLFW_TRUE : GLFW_FALSE);
        }
        firstMouse_ = true;
    }
    state.cursorCaptured = cursorCaptured_;

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(handle, &mouseX, &mouseY);
    if (firstMouse_) {
        lastX_ = mouseX;
        lastY_ = mouseY;
        firstMouse_ = false;
    }
    double xoffset = cursorCaptured_ ? (mouseX - lastX_) : 0.0;
    double yoffset = cursorCaptured_ ? (mouseY - lastY_) : 0.0;
    lastX_ = mouseX;
    lastY_ = mouseY;

    const float sensitivity = 0.08f;
    state.lookDelta.x = -static_cast<float>(xoffset) * sensitivity;
    state.lookDelta.y = static_cast<float>(yoffset) * sensitivity;

    auto keyDown = [&](int key) -> bool {
        return glfwGetKey(handle, key) == GLFW_PRESS;
    };

    state.moveForward = keyDown(GLFW_KEY_W);
    state.moveBackward = keyDown(GLFW_KEY_S);
    state.moveLeft = keyDown(GLFW_KEY_A);
    state.moveRight = keyDown(GLFW_KEY_D);
    state.ascend = keyDown(GLFW_KEY_SPACE);
    state.descend = keyDown(GLFW_KEY_LEFT_CONTROL) || keyDown(GLFW_KEY_RIGHT_CONTROL);
    state.boost = keyDown(GLFW_KEY_LEFT_SHIFT) || keyDown(GLFW_KEY_RIGHT_SHIFT);
    state.requestExit = keyDown(GLFW_KEY_ESCAPE);

    bool surfacePressed = keyDown(GLFW_KEY_F);
    state.toggleSurfaceWalk = surfacePressed && !surfacePrev_;
    surfacePrev_ = surfacePressed;

    bool thirdPressed = keyDown(GLFW_KEY_G);
    state.toggleThirdPerson = thirdPressed && !thirdPrev_;
    thirdPrev_ = thirdPressed;

    bool macroPressed = keyDown(GLFW_KEY_M);
    state.toggleMacroSkip = macroPressed && !macroPrev_;
    macroPrev_ = macroPressed;

    bool blockPressed = keyDown(GLFW_KEY_B);
    state.toggleBlockNormals = blockPressed && !blockPrev_;
    blockPrev_ = blockPressed;

    if (keyDown(GLFW_KEY_1)) state.debugPreset = 1u;
    if (keyDown(GLFW_KEY_2)) state.debugPreset = 2u;
    if (keyDown(GLFW_KEY_3)) state.debugPreset = 4u;
    if (keyDown(GLFW_KEY_4)) state.debugPreset = 16u;
    if (keyDown(GLFW_KEY_0)) state.debugPreset = 0u;
#else
    (void)window;
#endif
    return state;
}

}
