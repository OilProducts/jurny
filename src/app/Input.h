#pragma once

#include <glm/vec2.hpp>
#include <optional>

namespace platform {
class Window;
}

namespace app {

struct InputState {
    glm::vec2 lookDelta{0.0f};
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool ascend = false;
    bool descend = false;
    bool boost = false;
    bool toggleSurfaceWalk = false;
    bool toggleThirdPerson = false;
    bool toggleMacroSkip = false;
    bool toggleBlockNormals = false;
    bool requestExit = false;
    bool cursorCaptured = false;
    std::optional<uint32_t> debugPreset;
};

class Input {
public:
    void initialize(platform::Window& window);
    InputState sample(platform::Window& window);

private:
    bool cursorCaptured_ = false;
    bool firstMouse_ = true;
    bool rawMouseSupported_ = false;
    double lastX_ = 0.0;
    double lastY_ = 0.0;
    bool surfacePrev_ = false;
    bool thirdPrev_ = false;
    bool macroPrev_ = false;
    bool blockPrev_ = false;
};

}
