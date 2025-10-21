#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

#include "math/Spherical.h"

// WorldGen — procedural height field + caves for the spherical planet.
namespace world {

class WorldGen {
public:
    using NoiseParams = math::NoiseParams;

    void configure(const math::PlanetParams& planet,
                   const NoiseParams& noise,
                   std::uint32_t seed);

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

private:
    math::PlanetParams planet_{};
    NoiseParams params_{};
    std::uint32_t seed_ = 0;
};

}
