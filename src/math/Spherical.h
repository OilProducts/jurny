#pragma once

#include <glm/vec3.hpp>

// Spherical — planet math: F(p), gradF, ENU, shell intersection/overlap, gravity.
namespace math {
struct PlanetParams { double R{}, T{}, sea{}, Hmax{}; };

float  F_crust(const glm::vec3& p, const PlanetParams& P);
glm::vec3 gradF(const glm::vec3& p, const PlanetParams& P, float eps);
bool IntersectSphereShell(const glm::vec3& o, const glm::vec3& d, float Rin, float Rout,
                          float& tEnter, float& tExit);
void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up);
}

