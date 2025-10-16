#include "Spherical.h"

namespace math {
float F_crust(const glm::vec3&, const PlanetParams&) { return 0.0f; }
glm::vec3 gradF(const glm::vec3&, const PlanetParams&, float) { return glm::vec3(0.0f); }
bool IntersectSphereShell(const glm::vec3&, const glm::vec3&, float, float, float&, float&) { return false; }
void ENU(const glm::vec3&, glm::vec3&, glm::vec3&, glm::vec3&) {}
}

