#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

// Spherical — planet math: implicit crust field F(p), gradients, shell helpers, ENU frames.
namespace math {
struct PlanetParams { double R{}, T{}, sea{}, Hmax{}; };

struct NoiseParams {
    float continentFrequency = 0.04f;
    float continentAmplitude = 12.0f;
    int   continentOctaves   = 5;
    float detailFrequency    = 0.12f;
    float detailAmplitude    = 3.0f;
    int   detailOctaves      = 3;
    float warpFrequency      = 0.18f;
    float warpAmplitude      = 0.6f;
    float caveFrequency      = 0.35f;
    float caveAmplitude      = 4.0f;
    float caveThreshold      = 0.25f;
    float moistureFrequency  = 0.5f;
    int   moistureOctaves    = 3;

    static NoiseParams disabled();
};

inline constexpr int kNoiseCaveOctaves = 4;

struct CrustSample {
    float field = 0.0f;   // Signed distance-like value (neg = solid).
    float height = 0.0f;  // Height above base radius before caves (meters).
};

CrustSample SampleCrust(const glm::vec3& p, const PlanetParams& planet,
                        const NoiseParams& noise, std::uint32_t seed);

float  F_crust(const glm::vec3& p, const PlanetParams& planet,
               const NoiseParams& noise, std::uint32_t seed);
glm::vec3 gradF(const glm::vec3& p, const PlanetParams& planet,
                const NoiseParams& noise, std::uint32_t seed, float eps);

// Legacy helpers (sphere-only) retained for simplicity.
inline float F_crust(const glm::vec3& p, const PlanetParams& planet) {
    return F_crust(p, planet, NoiseParams::disabled(), 0u);
}
inline glm::vec3 gradF(const glm::vec3& p, const PlanetParams& planet, float eps) {
    return gradF(p, planet, NoiseParams::disabled(), 0u, eps);
}

bool IntersectSphereShell(const glm::vec3& o, const glm::vec3& d,
                          float Rin, float Rout,
                          float& tEnter, float& tExit);
void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up);

// Expose noise utilities so world generation can stay in sync with rendering.
glm::vec3 ApplyDomainWarp(const glm::vec3& unitDir,
                          const NoiseParams& noise, std::uint32_t seed);
float FractalBrownianMotion(const glm::vec3& p, float baseFrequency,
                            int octaves, float persistence,
                            std::uint32_t seed);
}
