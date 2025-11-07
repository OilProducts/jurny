#include "WorldGen.h"

#include <algorithm>
#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace world {

namespace {
constexpr float kMoisturePersistence = 0.55f;
constexpr glm::vec3 kMoistureOffset(23.1f, 91.7f, 44.4f);
} // namespace

void WorldGen::configure(const math::PlanetParams& planet,
                         const NoiseParams& noise,
                         std::uint32_t seed) {
    planet_ = planet;
    params_ = noise;
    seed_ = seed;
}

float WorldGen::crustField(const glm::vec3& p) const {
    return math::F_crust(p, planet_, params_, seed_);
}

WorldGen::BiomeSample WorldGen::biomeSample(const glm::vec3& p) const {
    BiomeSample sample{};
    const math::CrustSample crust = math::SampleCrust(p, planet_, params_, seed_);
    sample.height = crust.height;

    const float r = glm::length(p);
    const glm::vec3 dir = (r > 0.0f) ? (p / r) : glm::vec3(0.0f, 0.0f, 1.0f);
    const float baseRadius = static_cast<float>(planet_.R);
    const glm::vec3 surfacePoint = dir * baseRadius;
    const glm::vec3 warped = math::ApplyDomainWarp(surfacePoint, params_, seed_);

    const int moistOct = std::max(params_.moistureOctaves, 1);
    const float baseFreq = params_.moistureFrequency;
    float moisture = 0.0f;
    if (baseFreq > 0.0f && moistOct > 0) {
        moisture = math::FractalBrownianMotion(
            warped * baseFreq + kMoistureOffset,
            baseFreq,
            moistOct,
            kMoisturePersistence,
            seed_ + 877u);
    }
    sample.moisture = glm::clamp(moisture * 0.5f + 0.5f, 0.0f, 1.0f);
    sample.temperature = glm::clamp(1.0f - std::abs(dir.z), 0.0f, 1.0f);
    return sample;
}

glm::vec3 WorldGen::crustNormal(const glm::vec3& p, float eps) const {
    return math::gradF(p, planet_, params_, seed_, eps);
}

}
