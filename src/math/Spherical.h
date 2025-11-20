#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

// Spherical — planet math: implicit crust field F(p), gradients, shell helpers, ENU frames.
namespace math {
struct PlanetParams { double R{}, T{}, sea{}, Hmax{}; };

struct NoiseParams {
    float macroFrequency = 0.0015f;   // cycles per meter on the surface
    float macroAmplitude = 120.0f;    // meters
    float macroRidgeWeight = 0.35f;   // mix between base and ridged macro noise
    float macroSharpness = 1.3f;      // ridged exponent

    float detailFrequency = 0.04f;    // cycles per meter
    float detailAmplitude = 10.0f;    // meters
    float detailRidgeWeight = 0.25f;  // mix between base/detail ridges
    float detailSharpness = 1.0f;     // ridged exponent

    float bandFrequency = 3.0f;       // stripes per hemisphere
    float bandAmplitude = 24.0f;      // meters contributed by latitude bands
    float bandSharpness = 1.4f;       // falloff from equator
    float baseHeightOffset = 0.0f;    // meters added after layering

    float cavityFrequency = 0.25f;    // cycles per meter in 3D space
    float cavityAmplitude = 5.0f;     // meters carved when mask triggers
    float cavityThreshold = 0.35f;    // [0,1] threshold for cavities
    float cavityContrast = 2.0f;      // steeper falloff -> sharper caves

    float moistureFrequency = 0.002f; // cycles per meter for biome sampling
    int   moistureOctaves   = 3;
    float moisturePersistence = 0.5f;
    float padding = 0.0f;             // reserved / alignment

    static NoiseParams disabled();
};

struct NoiseTuning {
    float macroWavelength   = 2500.0f; // meters per macro cycle
    float macroAmplitude    = 120.0f;  // meters
    float macroRidgeWeight  = 0.35f;   // [0,1]
    float macroSharpness    = 1.3f;    // >=1 for sharper ridges

    float detailWavelength  = 180.0f;  // meters per detail cycle
    float detailAmplitude   = 10.0f;   // meters
    float detailRidgeWeight = 0.25f;
    float detailSharpness   = 1.0f;

    float bandCount         = 3.0f;    // stripes between pole->pole
    float bandAmplitude     = 24.0f;   // meters
    float bandSharpness     = 1.4f;    // falloff from equator
    float baseHeightOffset  = 0.0f;    // meters

    float caveWavelength    = 32.0f;   // meters per cavity repetition
    float caveAmplitude     = 5.0f;    // meters carved
    float caveThreshold     = 0.35f;   // [0,1]
    float caveContrast      = 2.0f;    // slope of activation curve

    float moistureWavelength = 900.0f; // meters per biome noise cycle
    int   moistureOctaves    = 4;
    float moisturePersistence = 0.55f;
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
