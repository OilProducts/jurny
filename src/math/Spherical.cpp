#include "Spherical.h"
#include <glm/geometric.hpp>
#include <algorithm>
#include <cmath>

namespace math {
namespace {
// Solve |o + t d|^2 = R^2. Allows non-normalized d. Returns t0<=t1 on hit.
inline bool intersectSphere(const glm::vec3& o, const glm::vec3& d, float R,
                            float& t0, float& t1) {
    const float a = glm::dot(d, d);
    if (a <= 0.0f) return false; // degenerate direction
    const float b = glm::dot(o, d);              // note: quadratic uses 2b, we keep b and adjust disc
    const float c = glm::dot(o, o) - R * R;
    const float disc = b * b - a * c;           // discriminant of a t^2 + 2b t + c = 0
    if (disc < 0.0f) return false;
    const float s = std::sqrt(std::max(disc, 0.0f));
    const float invA = 1.0f / a;
    float tnear = (-b - s) * invA;
    float tfar  = (-b + s) * invA;
    if (tnear > tfar) std::swap(tnear, tfar);
    t0 = tnear; t1 = tfar;
    return true;
}
} // namespace

// Base continuous crust field (negative = solid). For M0 we implement a smooth sphere
// with radius R: F(p) = |p| - R. Height/caves can be layered later without changing the API.
float F_crust(const glm::vec3& p, const PlanetParams& P) {
    const float r = glm::length(p);
    return r - static_cast<float>(P.R);
}

// Gradient of F. For the base sphere, grad is simply p/|p|. The eps parameter is reserved
// for future finite-difference evaluation when F includes noise/warps.
glm::vec3 gradF(const glm::vec3& p, const PlanetParams&, float /*eps*/) {
    const float r = glm::length(p);
    if (r > 0.0f) return p * (1.0f / r);
    // Arbitrary up when at center (shouldn’t happen for surface hits)
    return glm::vec3(0.0f, 1.0f, 0.0f);
}

// Intersect a ray with the spherical shell Rin..Rout (outside Rin, inside Rout).
// Returns the nearest non-empty positive segment [tEnter, tExit) in shell space.
bool IntersectSphereShell(const glm::vec3& o, const glm::vec3& d,
                          float Rin, float Rout,
                          float& tEnter, float& tExit) {
    // Preconditions
    if (!(Rout > Rin && Rin >= 0.0f)) return false;

    float to0, to1; // outer sphere interval
    if (!intersectSphere(o, d, Rout, to0, to1)) return false;

    // Clamp to t >= 0 (we only care about forward intersections)
    const float eps = 1e-6f;
    const float outerStart = std::max(to0, eps);
    const float outerEnd   = to1;
    if (outerEnd <= outerStart) return false;

    float ti0, ti1; // inner sphere interval
    const bool hitInner = intersectSphere(o, d, Rin, ti0, ti1);

    if (!hitInner) {
        tEnter = outerStart; tExit = outerEnd;
        return tExit > tEnter;
    }

    // Shell intervals are [outerStart, min(outerEnd, ti0)] and [max(outerStart, ti1), outerEnd]
    const float c1Start = outerStart;
    const float c1End   = std::min(outerEnd, ti0);
    if (c1End > c1Start) { tEnter = c1Start; tExit = c1End; return true; }

    const float c2Start = std::max(outerStart, ti1);
    const float c2End   = outerEnd;
    if (c2End > c2Start) { tEnter = c2Start; tExit = c2End; return true; }

    return false;
}

void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up) {
    up = glm::normalize(p);
    // Reference axis for east: global Z. Handle polar degeneracy.
    const glm::vec3 z(0.0f, 0.0f, 1.0f);
    east = glm::normalize(glm::cross(z, up));
    if (glm::dot(east, east) < 1e-6f) east = glm::vec3(1.0f, 0.0f, 0.0f);
    north = glm::cross(up, east);
}
}
