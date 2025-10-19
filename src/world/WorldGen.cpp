#include "WorldGen.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>

namespace world {

namespace {
constexpr float kPI = glm::pi<float>();
constexpr float kPersistenceContinent = 0.55f;
constexpr float kPersistenceDetail    = 0.5f;
constexpr float kCavePersistence      = 0.5f;
}

void WorldGen::configure(const math::PlanetParams& planet, const NoiseParams& noise, std::uint32_t seed) {
    planet_ = planet;
    params_ = noise;
    seed_ = seed;
}

float WorldGen::fade(float t) {
    return t * t * (3.0f - 2.0f * t);
}

glm::vec3 WorldGen::fade(const glm::vec3& v) {
    return glm::vec3(fade(v.x), fade(v.y), fade(v.z));
}

float WorldGen::hash(const glm::vec3& p, std::uint32_t seed) {
    const glm::vec3 kHashVec(12.9898f, 78.233f, 37.719f);
    float n = glm::dot(p, kHashVec) + static_cast<float>(seed) * 0.001953125f; // /512
    return glm::fract(glm::sin(n) * 43758.5453f);
}

float WorldGen::valueNoise(const glm::vec3& p, std::uint32_t seedOffset) const {
    glm::vec3 i = glm::floor(p);
    glm::vec3 f = glm::fract(p);
    glm::vec3 u = fade(f);

    auto corner = [&](int ox, int oy, int oz) {
        glm::vec3 offset(static_cast<float>(ox), static_cast<float>(oy), static_cast<float>(oz));
        return hash(i + offset, seed_ + seedOffset + static_cast<std::uint32_t>(ox + oy * 17 + oz * 131));
    };

    float n000 = corner(0,0,0);
    float n100 = corner(1,0,0);
    float n010 = corner(0,1,0);
    float n110 = corner(1,1,0);
    float n001 = corner(0,0,1);
    float n101 = corner(1,0,1);
    float n011 = corner(0,1,1);
    float n111 = corner(1,1,1);

    float nx00 = glm::mix(n000, n100, u.x);
    float nx10 = glm::mix(n010, n110, u.x);
    float nx01 = glm::mix(n001, n101, u.x);
    float nx11 = glm::mix(n011, n111, u.x);
    float nxy0 = glm::mix(nx00, nx10, u.y);
    float nxy1 = glm::mix(nx01, nx11, u.y);
    return glm::mix(nxy0, nxy1, u.z);
}

float WorldGen::fbm(const glm::vec3& p, float baseFrequency, int octaves, float persistence, std::uint32_t seedOffset) const {
    float amplitude = 1.0f;
    float frequency = baseFrequency;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude * valueNoise(p * frequency, seedOffset + static_cast<std::uint32_t>(i) * 97u);
        norm += amplitude;
        frequency *= 2.0f;
        amplitude *= persistence;
    }
    if (norm <= 0.0f) return 0.0f;
    float result = sum / norm;
    return result * 2.0f - 1.0f; // [-1,1]
}

glm::vec3 WorldGen::domainWarp(const glm::vec3& p) const {
    if (params_.warpAmplitude <= 0.0f || params_.warpFrequency <= 0.0f) return p;
    int detailOct = std::max(params_.detailOctaves, 1);
    float fx = fbm(p + glm::vec3(31.7f, 17.3f, 13.1f), params_.warpFrequency, detailOct, kPersistenceDetail, seed_ + 233u);
    float fy = fbm(p + glm::vec3(11.1f, 53.2f, 27.8f), params_.warpFrequency, detailOct, kPersistenceDetail, seed_ + 389u);
    float fz = fbm(p + glm::vec3(91.7f, 45.3f, 67.1f), params_.warpFrequency, detailOct, kPersistenceDetail, seed_ + 521u);
    glm::vec3 warp(fx, fy, fz);
    return p + warp * params_.warpAmplitude;
}

void WorldGen::evaluate(const glm::vec3& p, float& field, float& height) const {
    float r = glm::length(p);
    if (r <= 0.0f) {
        height = -static_cast<float>(planet_.T);
        field = -static_cast<float>(planet_.T);
        return;
    }

    glm::vec3 dir = p / r;
    glm::vec3 warped = domainWarp(dir);
    int contOct = std::max(params_.continentOctaves, 1);
    int detailOct = std::max(params_.detailOctaves, 1);
    float continents = fbm(warped, params_.continentFrequency, contOct, kPersistenceContinent, seed_ + 0u);
    float detail = fbm(warped * 2.0f, params_.detailFrequency, detailOct, kPersistenceDetail, seed_ + 613u);
    height = params_.continentAmplitude * continents + params_.detailAmplitude * detail;
    height = std::clamp(height, -static_cast<float>(planet_.T), static_cast<float>(planet_.Hmax));
    float surfaceRadius = static_cast<float>(planet_.R) + height;
    field = r - surfaceRadius;
    if (field < 0.0f && params_.caveAmplitude > 0.0f) {
        int caveOct = std::max(WorldGen::kCaveOctaves, 1);
        float cave = fbm(p, params_.caveFrequency, caveOct, kCavePersistence, seed_ + 997u);
        float cavity = cave - params_.caveThreshold;
        if (cavity > 0.0f) field += -params_.caveAmplitude * cavity;
    }
}

float WorldGen::crustField(const glm::vec3& p) const {
    float field, height;
    evaluate(p, field, height);
    return field;
}

WorldGen::BiomeSample WorldGen::biomeSample(const glm::vec3& p) const {
    BiomeSample sample{};
    float field, height;
    evaluate(p, field, height);
    sample.height = height;
    float r = glm::length(p);
    glm::vec3 dir = (r > 0.0f) ? (p / r) : glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 warped = domainWarp(dir);
    int moistOct = std::max(params_.moistureOctaves, 1);
    float moisture = fbm(warped * params_.moistureFrequency + glm::vec3(23.1f, 91.7f, 44.4f),
                         params_.moistureFrequency, moistOct, 0.55f, seed_ + 877u);
    sample.moisture = glm::clamp(moisture * 0.5f + 0.5f, 0.0f, 1.0f);
    sample.temperature = glm::clamp(1.0f - std::abs(dir.z), 0.0f, 1.0f);
    return sample;
}

glm::vec3 WorldGen::crustNormal(const glm::vec3& p, float eps) const {
    glm::vec3 dx(eps, 0.0f, 0.0f);
    glm::vec3 dy(0.0f, eps, 0.0f);
    glm::vec3 dz(0.0f, 0.0f, eps);
    float fx1 = crustField(p + dx);
    float fx0 = crustField(p - dx);
    float fy1 = crustField(p + dy);
    float fy0 = crustField(p - dy);
    float fz1 = crustField(p + dz);
    float fz0 = crustField(p - dz);
    glm::vec3 g((fx1 - fx0), (fy1 - fy0), (fz1 - fz0));
    if (glm::dot(g, g) < 1e-12f) return glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::normalize(g);
}

}
