#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

// Spherical — planet math: implicit crust field F(p), gradients, shell helpers, ENU frames.
namespace math {
struct PlanetParams { double R{}, T{}, sea{}, Hmax{}; };

struct NoiseParams {
    float continentFrequency = 0.04f;   // cycles per meter
    float continentAmplitude = 12.0f;   // meters
    int   continentOctaves   = 5;
    float detailFrequency    = 0.12f;   // cycles per meter
    float detailAmplitude    = 3.0f;    // meters
    int   detailOctaves      = 3;
    float detailWarpMultiplier = 2.0f;  // scales the detail domain
    float baseHeightOffset   = 0.0f;    // meters added after noise
    float warpFrequency      = 0.18f;   // cycles per meter
    float warpAmplitude      = 0.6f;    // meters
    float slopeSampleDistance = 20.0f;  // meters along surface for slope mask
    float caveFrequency      = 0.35f;   // cycles per meter (world space)
    float caveAmplitude      = 4.0f;    // meters
    float caveThreshold      = 0.25f;
    float moistureFrequency  = 0.5f;    // cycles per meter
    int   moistureOctaves    = 3;

    static NoiseParams disabled();
};

inline constexpr int kNoiseCaveOctaves = 4;

struct NoiseTuning {
    float continentsPerCircumference = 4.0f; // approximate count around equator
    float continentAmplitude = 120.0f;       // meters
    int   continentOctaves   = 6;
    float detailWavelength   = 32.0f;        // meters
    float detailAmplitude    = 18.0f;        // meters
    int   detailOctaves      = 4;
    float detailWarpMultiplier = 2.5f;
    float baseHeightOffset   = 0.0f;         // meters
    float warpWavelength     = 140.0f;       // meters
    float warpAmplitude      = 45.0f;        // meters
    float slopeSampleDistance = 48.0f;       // meters
    float caveWavelength     = 18.0f;        // meters
    float caveAmplitude      = 6.0f;         // meters
    float caveThreshold      = 0.35f;
    float moistureWavelength = 900.0f;       // meters
    int   moistureOctaves    = 4;
};

NoiseParams BuildNoiseParams(const NoiseTuning& tuning, const PlanetParams& planet);

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
// Apply shared domain warp in world-space (meters). Expects the input position
// to lie on the planet surface (normalized direction * radius).
glm::vec3 ApplyDomainWarp(const glm::vec3& surfacePoint,
                          const NoiseParams& noise, std::uint32_t seed);
float FractalBrownianMotion(const glm::vec3& p, float baseFrequency,
                            int octaves, float persistence,
                            std::uint32_t seed);
}
