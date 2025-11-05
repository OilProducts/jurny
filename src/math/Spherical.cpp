#include "Spherical.h"

#include <algorithm>
#include <cmath>
#include <glm/common.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

namespace math {
namespace {
constexpr float kPersistenceContinent = 0.55f;
constexpr float kPersistenceDetail    = 0.5f;
constexpr float kPersistenceCave      = 0.5f;

inline std::uint32_t pcgHash32(std::uint32_t v) {
    v = v * 747796405u + 2891336453u;
    const std::uint32_t word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

inline std::uint32_t latticeHash(const glm::ivec3& c, std::uint32_t seed) {
    std::uint32_t v = pcgHash32(static_cast<std::uint32_t>(c.x) ^ seed);
    v = pcgHash32(static_cast<std::uint32_t>(c.y) ^ v);
    v = pcgHash32(static_cast<std::uint32_t>(c.z) ^ v);
    return v;
}

inline float fade(float t) {
    return t * t * (3.0f - 2.0f * t);
}

inline glm::vec3 fade3(const glm::vec3& v) {
    return glm::vec3(fade(v.x), fade(v.y), fade(v.z));
}

inline float hash(const glm::vec3& p, std::uint32_t seed) {
    constexpr glm::vec3 kHashVec(12.9898f, 78.233f, 37.719f);
    const float n = glm::dot(p, kHashVec) + static_cast<float>(seed) * 0.001953125f; // /512
    return glm::fract(std::sin(n) * 43758.5453f);
}

float valueNoise(const glm::vec3& p, std::uint32_t seed) {
    const glm::vec3 i = glm::floor(p);
    const glm::vec3 f = glm::fract(p);
    const glm::vec3 u = fade3(f);
    const glm::ivec3 cell = glm::ivec3(i);

    const auto corner = [&](int ox, int oy, int oz) {
        const glm::vec3 offset(static_cast<float>(ox),
                               static_cast<float>(oy),
                               static_cast<float>(oz));
        const glm::ivec3 cornerIdx = cell + glm::ivec3(ox, oy, oz);
        const std::uint32_t cornerSeed = latticeHash(cornerIdx, seed);
        const glm::vec3 pos = i + offset;
        return hash(pos, cornerSeed);
    };

    const float n000 = corner(0, 0, 0);
    const float n100 = corner(1, 0, 0);
    const float n010 = corner(0, 1, 0);
    const float n110 = corner(1, 1, 0);
    const float n001 = corner(0, 0, 1);
    const float n101 = corner(1, 0, 1);
    const float n011 = corner(0, 1, 1);
    const float n111 = corner(1, 1, 1);

    const float nx00 = glm::mix(n000, n100, u.x);
    const float nx10 = glm::mix(n010, n110, u.x);
    const float nx01 = glm::mix(n001, n101, u.x);
    const float nx11 = glm::mix(n011, n111, u.x);
    const float nxy0 = glm::mix(nx00, nx10, u.y);
    const float nxy1 = glm::mix(nx01, nx11, u.y);
    return glm::mix(nxy0, nxy1, u.z);
}

float fbmInternal(const glm::vec3& p,
                  float baseFrequency,
                  int octaves,
                  float persistence,
                  std::uint32_t seed) {
    if (baseFrequency <= 0.0f || octaves <= 0) {
        return 0.0f;
    }
    float amplitude = 1.0f;
    float frequency = baseFrequency;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        const std::uint32_t octaveSeed =
            seed + static_cast<std::uint32_t>(i) * 97u;
        sum += amplitude * valueNoise(p * frequency, octaveSeed);
        norm += amplitude;
        frequency *= 2.0f;
        amplitude *= persistence;
    }
    if (norm <= 0.0f) return 0.0f;
    const float result = sum / norm;
    return result * 2.0f - 1.0f; // [-1, 1]
}

glm::vec3 applyDomainWarpInternal(const glm::vec3& dir,
                                  const NoiseParams& noise,
                                  std::uint32_t seed) {
    if (noise.warpAmplitude <= 0.0f || noise.warpFrequency <= 0.0f) {
        return dir;
    }
    const int detailOct = std::max(noise.detailOctaves, 1);
    const glm::vec3 offX(31.7f, 17.3f, 13.1f);
    const glm::vec3 offY(11.1f, 53.2f, 27.8f);
    const glm::vec3 offZ(91.7f, 45.3f, 67.1f);
    const float fx = fbmInternal(dir + offX, noise.warpFrequency,
                                 detailOct, kPersistenceDetail,
                                 seed + 233u);
    const float fy = fbmInternal(dir + offY, noise.warpFrequency,
                                 detailOct, kPersistenceDetail,
                                 seed + 389u);
    const float fz = fbmInternal(dir + offZ, noise.warpFrequency,
                                 detailOct, kPersistenceDetail,
                                 seed + 521u);
    const glm::vec3 warp(fx, fy, fz);
    return dir + warp * noise.warpAmplitude;
}

CrustSample evaluateCrust(const glm::vec3& p,
                          const PlanetParams& planet,
                          const NoiseParams& noise,
                          std::uint32_t seed) {
    CrustSample sample{};

    const float r = glm::length(p);
    if (r <= 0.0f) {
        const float floor = -static_cast<float>(planet.T);
        sample.field = floor;
        sample.height = floor;
        return sample;
    }

    const glm::vec3 dir = p / r;
    const glm::vec3 warped = applyDomainWarpInternal(dir, noise, seed);

    const int contOct = std::max(noise.continentOctaves, 1);
    const int detailOct = std::max(noise.detailOctaves, 1);
    const float continents = fbmInternal(
        warped,
        noise.continentFrequency,
        contOct,
        kPersistenceContinent,
        seed);
    const float detail = fbmInternal(
        warped * 2.0f,
        noise.detailFrequency,
        detailOct,
        kPersistenceDetail,
        seed + 613u);

    float continentHeight = noise.continentAmplitude * continents;

    float slopeMask = 0.0f;
    if (noise.continentAmplitude > 0.0f && noise.continentFrequency > 0.0f) {
        glm::vec3 east, north, upVec;
        ENU(dir, east, north, upVec);
        const float gradStep = 0.02f;
        auto sampleContinents = [&](const glm::vec3& w) {
            return fbmInternal(w,
                               noise.continentFrequency,
                               contOct,
                               kPersistenceContinent,
                               seed);
        };
        float continentsEast = sampleContinents(warped + east * gradStep);
        float continentsNorth = sampleContinents(warped + north * gradStep);
        float slope = glm::length(glm::vec2(continentsEast - continents,
                                            continentsNorth - continents)) / gradStep;
        slopeMask = glm::smoothstep(0.3f, 1.2f, slope);
    }

    float detailMask = 1.0f - glm::smoothstep(0.55f, 0.95f, continents * 0.5f + 0.5f);
    float detailStrength = glm::mix(1.0f, 0.25f, slopeMask);
    float detailContribution = noise.detailAmplitude * detailMask * detailStrength * detail;
    float height = continentHeight + detailContribution;
    if (slopeMask > 0.0f) {
        const float slopeFlatten = glm::mix(0.0f, noise.continentAmplitude * 0.3f, slopeMask);
        height = glm::mix(height, height - slopeFlatten, slopeMask);
    }

    auto smoothClamp = [](float value, float lo, float hi, float transition) {
        if (transition <= 0.0f || hi <= lo) {
            return glm::clamp(value, lo, hi);
        }
        float tLo = glm::smoothstep(lo - transition, lo + transition, value);
        float vLo = glm::mix(lo, value, tLo);
        float tHi = glm::smoothstep(hi - transition, hi + transition, value);
        return glm::mix(vLo, hi, tHi);
    };

    const float minHeight = -static_cast<float>(planet.T);
    const float maxHeight = static_cast<float>(planet.Hmax);
    const float plateau =
        std::max(12.0f,
                 std::max(static_cast<float>(planet.T) * 0.35f,
                          static_cast<float>(planet.Hmax) * 0.4f));
    height = smoothClamp(height, minHeight, maxHeight, plateau);

    const float surfaceRadius = static_cast<float>(planet.R) + height;
    float field = r - surfaceRadius;

    if (field < 0.0f && noise.caveAmplitude > 0.0f &&
        noise.caveFrequency > 0.0f) {
        const int caveOct = std::max(kNoiseCaveOctaves, 1);
        const float cave = fbmInternal(
            p,
            noise.caveFrequency,
            caveOct,
            kPersistenceCave,
            seed + 997u);
        const float cavity = cave - noise.caveThreshold;
        if (cavity > 0.0f) {
            field += -noise.caveAmplitude * cavity;
        }
    }

    sample.field = field;
    sample.height = height;
    return sample;
}

// Solve |o + t d|^2 = R^2. Allows non-normalized d. Returns t0<=t1 on hit.
inline bool intersectSphere(const glm::vec3& o, const glm::vec3& d, float R,
                            float& t0, float& t1) {
    const float a = glm::dot(d, d);
    if (a <= 0.0f) return false; // degenerate direction
    const float b = glm::dot(o, d);              // quadratic uses 2b; keep b and adjust disc
    const float c = glm::dot(o, o) - R * R;
    const float disc = b * b - a * c;
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

NoiseParams NoiseParams::disabled() {
    NoiseParams n{};
    n.continentFrequency = 0.0f;
    n.continentAmplitude = 0.0f;
    n.continentOctaves   = 0;
    n.detailFrequency    = 0.0f;
    n.detailAmplitude    = 0.0f;
    n.detailOctaves      = 0;
    n.warpFrequency      = 0.0f;
    n.warpAmplitude      = 0.0f;
    n.caveFrequency      = 0.0f;
    n.caveAmplitude      = 0.0f;
    n.caveThreshold      = 1.0f;
    n.moistureFrequency  = 0.0f;
    n.moistureOctaves    = 0;
    return n;
}

glm::vec3 ApplyDomainWarp(const glm::vec3& unitDir,
                          const NoiseParams& noise,
                          std::uint32_t seed) {
    return applyDomainWarpInternal(unitDir, noise, seed);
}

float FractalBrownianMotion(const glm::vec3& p, float baseFrequency,
                            int octaves, float persistence,
                            std::uint32_t seed) {
    return fbmInternal(p, baseFrequency, octaves, persistence, seed);
}

CrustSample SampleCrust(const glm::vec3& p, const PlanetParams& planet,
                        const NoiseParams& noise, std::uint32_t seed) {
    return evaluateCrust(p, planet, noise, seed);
}

float F_crust(const glm::vec3& p, const PlanetParams& planet,
              const NoiseParams& noise, std::uint32_t seed) {
    return evaluateCrust(p, planet, noise, seed).field;
}

glm::vec3 gradF(const glm::vec3& p, const PlanetParams& planet,
                const NoiseParams& noise, std::uint32_t seed, float eps) {
    float step = eps;
    if (step <= 0.0f) {
        step = 0.5f; // fallback; caller should provide voxel-scale epsilon
    }
    const glm::vec3 dx(step, 0.0f, 0.0f);
    const glm::vec3 dy(0.0f, step, 0.0f);
    const glm::vec3 dz(0.0f, 0.0f, step);

    const float fx1 = F_crust(p + dx, planet, noise, seed);
    const float fx0 = F_crust(p - dx, planet, noise, seed);
    const float fy1 = F_crust(p + dy, planet, noise, seed);
    const float fy0 = F_crust(p - dy, planet, noise, seed);
    const float fz1 = F_crust(p + dz, planet, noise, seed);
    const float fz0 = F_crust(p - dz, planet, noise, seed);

    glm::vec3 g(fx1 - fx0, fy1 - fy0, fz1 - fz0);
    if (glm::dot(g, g) <= 0.0f) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return glm::normalize(g);
}

bool IntersectSphereShell(const glm::vec3& o, const glm::vec3& d,
                          float Rin, float Rout,
                          float& tEnter, float& tExit) {
    if (!(Rout > Rin && Rin >= 0.0f)) return false;

    float to0, to1;
    if (!intersectSphere(o, d, Rout, to0, to1)) return false;

    const float eps = 1e-6f;
    const float outerStart = std::max(to0, eps);
    const float outerEnd   = to1;
    if (outerEnd <= outerStart) return false;

    float ti0, ti1;
    const bool hitInner = intersectSphere(o, d, Rin, ti0, ti1);

    if (!hitInner) {
        tEnter = outerStart;
        tExit  = outerEnd;
        return tExit > tEnter;
    }

    const float c1Start = outerStart;
    const float c1End   = std::min(outerEnd, ti0);
    if (c1End > c1Start) {
        tEnter = c1Start;
        tExit  = c1End;
        return true;
    }

    const float c2Start = std::max(outerStart, ti1);
    const float c2End   = outerEnd;
    if (c2End > c2Start) {
        tEnter = c2Start;
        tExit  = c2End;
        return true;
    }

    return false;
}

void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up) {
    up = glm::normalize(p);
    const glm::vec3 z(0.0f, 0.0f, 1.0f);
    east = glm::normalize(glm::cross(z, up));
    if (glm::dot(east, east) < 1e-6f) {
        east = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    north = glm::cross(up, east);
}
} // namespace math
