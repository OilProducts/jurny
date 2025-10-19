#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

#include "math/Spherical.h"

// WorldGen — procedural height field + caves for the spherical planet.
namespace world {

class WorldGen {
public:
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
        int   moistureOctaves     = 3;
    };

    void configure(const math::PlanetParams& planet, const NoiseParams& noise, std::uint32_t seed);

    float crustField(const glm::vec3& p) const;

    struct BiomeSample {
        float height = 0.0f;      // meters above sea level
        float moisture = 0.0f;    // [0,1]
        float temperature = 0.0f; // [0,1], 1=warm equator
    };

    BiomeSample biomeSample(const glm::vec3& p) const;
    glm::vec3 crustNormal(const glm::vec3& p, float eps) const;

    const NoiseParams& params() const { return params_; }
    std::uint32_t seed() const { return seed_; }
    const math::PlanetParams& planet() const { return planet_; }
    static constexpr int kCaveOctaves = 4;

private:
    math::PlanetParams planet_{};
    NoiseParams params_{};
    std::uint32_t seed_ = 0;

    static float fade(float t);
    static glm::vec3 fade(const glm::vec3& v);
    static float hash(const glm::vec3& p, std::uint32_t seed);
    float valueNoise(const glm::vec3& p, std::uint32_t seedOffset) const;
    float fbm(const glm::vec3& p, float baseFrequency, int octaves, float persistence, std::uint32_t seedOffset) const;
    glm::vec3 domainWarp(const glm::vec3& p) const;
    void evaluate(const glm::vec3& p, float& field, float& height) const;
};

}
