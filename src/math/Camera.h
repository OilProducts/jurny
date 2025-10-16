#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// Camera — prev/curr matrices, jitter, and origin rebase bookkeeping.
namespace math {
struct Camera {
    glm::mat4 view{}, proj{}, prevView{}, prevProj{};
    glm::vec3 position{}; // local (rebased)
};
}

