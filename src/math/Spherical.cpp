#include "Spherical.h"

#include <algorithm>
#include <cmath>
#include <glm/common.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

namespace math {
namespace {

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

inline float randomFloat(std::uint32_t seed) {
    constexpr float kInvUint32 = 1.0f / 4294967296.0f;
    seed = pcgHash32(seed);
    return static_cast<float>(seed) * kInvUint32;
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
        return randomFloat(cornerSeed);
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

inline float signedValueNoise(const glm::vec3& p, std::uint32_t seed) {
    return valueNoise(p, seed) * 2.0f - 1.0f;
}

float ridgeSignal(float v, float sharpness) {
    float ridge = 1.0f - std::abs(v);
    ridge = glm::clamp(ridge, 0.0f, 1.0f);
    if (sharpness > 0.0f) {
        ridge = std::pow(ridge, sharpness);
    }
    return ridge * 2.0f - 1.0f;
}

float macroLayer(const glm::vec3& surfacePoint,
                 const glm::vec3& dir,
                 const NoiseParams& noise,
                 std::uint32_t seed) {
    if (noise.macroAmplitude <= 0.0f || noise.macroFrequency <= 0.0f) {
        return 0.0f;
    }
    const glm::vec3 domain = surfacePoint * noise.macroFrequency;
    const float base = signedValueNoise(domain, seed);
    const float ridgeSrc = signedValueNoise(domain + glm::vec3(19.1f, 7.7f, 13.3f), seed + 17u);
    float ridge = ridgeSignal(ridgeSrc, std::max(noise.macroSharpness, 0.5f));
    const float jitter = signedValueNoise(domain * 0.5f + glm::vec3(31.7f, 23.1f, 11.9f), seed + 31u);
    float macro = glm::mix(base, ridge, glm::clamp(noise.macroRidgeWeight, 0.0f, 1.0f));
    // Encourage polar flattening slightly by blending in |z| of the normal.
    const float polar = std::abs(dir.z) * 0.5f;
    macro = glm::mix(macro, jitter, 0.15f) + polar * 0.05f;
    return noise.macroAmplitude * macro;
}

float detailLayer(const glm::vec3& surfacePoint,
                  const NoiseParams& noise,
                  std::uint32_t seed) {
    if (noise.detailAmplitude <= 0.0f || noise.detailFrequency <= 0.0f) {
        return 0.0f;
    }
    const glm::vec3 domain = surfacePoint * noise.detailFrequency;
    const float base = signedValueNoise(domain + glm::vec3(3.1f, 7.3f, 11.9f), seed + 101u);
    const float ridgeSrc = signedValueNoise(domain * 1.7f + glm::vec3(17.0f, 5.0f, 9.0f), seed + 131u);
    float ridge = ridgeSignal(ridgeSrc, std::max(noise.detailSharpness, 0.5f));
    float trig = std::sin(glm::dot(domain, glm::vec3(0.8f, 1.1f, 0.5f)) + 1.3f) *
                 std::cos(glm::dot(domain, glm::vec3(1.5f, 0.4f, 1.7f)) - 0.6f);
    trig = glm::clamp(trig, -1.0f, 1.0f);
    float detail = glm::mix(base, ridge, glm::clamp(noise.detailRidgeWeight, 0.0f, 1.0f));
    detail = glm::mix(detail, trig, 0.25f);
    return noise.detailAmplitude * detail;
}

float latitudeBands(const glm::vec3& dir, const NoiseParams& noise) {
    if (noise.bandAmplitude <= 0.0f || noise.bandFrequency <= 0.0f) {
        return 0.0f;
    }
    const float lat = dir.z * 0.5f + 0.5f; // [0,1]
    float s = std::sin(lat * noise.bandFrequency * glm::two_pi<float>());
    const float falloff = std::exp(-noise.bandSharpness * (1.0f - std::abs(dir.z)));
    return noise.bandAmplitude * s * falloff;
}

float cavesContribution(const glm::vec3& p, const NoiseParams& noise, std::uint32_t seed) {
    if (noise.cavityAmplitude <= 0.0f || noise.cavityFrequency <= 0.0f) {
        return 0.0f;
    }
    const glm::vec3 domain = p * noise.cavityFrequency;
    const float base = signedValueNoise(domain, seed + 997u);
    const float pocket = signedValueNoise(domain + glm::vec3(13.0f, 29.0f, 17.0f), seed + 1019u);
    float signal = glm::mix(base, pocket, 0.5f);
    signal = signal * 0.5f + 0.5f; // [0,1]
    const float threshold = glm::clamp(noise.cavityThreshold, 0.0f, 1.0f);
    const float contrast = std::max(noise.cavityContrast, 0.01f);
    float mask = glm::clamp((signal - threshold) * contrast, 0.0f, 1.0f);
    mask = mask * mask;
    return -noise.cavityAmplitude * mask;
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
    const float baseRadius = static_cast<float>(planet.R);
    const glm::vec3 surfacePoint = dir * baseRadius;

    float height = noise.baseHeightOffset;
    height += macroLayer(surfacePoint, dir, noise, seed);
    height += detailLayer(surfacePoint, noise, seed);
    height += latitudeBands(dir, noise);

    const float minHeight = -static_cast<float>(planet.T);
    const float maxHeight = static_cast<float>(planet.Hmax);
    height = glm::clamp(height, minHeight, maxHeight);

    float field = r - (baseRadius + height);
    if (field < 0.0f) {
        field += cavesContribution(p, noise, seed);
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

NoiseParams BuildNoiseParams(const NoiseTuning& tuning, const PlanetParams& planet) {
    (void)planet;
    NoiseParams params{};

    auto wavelengthToFrequency = [](float wavelengthMeters) -> float {
        return (wavelengthMeters > 0.0f) ? (1.0f / wavelengthMeters) : 0.0f;
    };

    params.macroFrequency   = wavelengthToFrequency(tuning.macroWavelength);
    params.macroAmplitude   = tuning.macroAmplitude;
    params.macroRidgeWeight = glm::clamp(tuning.macroRidgeWeight, 0.0f, 1.0f);
    params.macroSharpness   = std::max(tuning.macroSharpness, 0.25f);

    params.detailFrequency   = wavelengthToFrequency(tuning.detailWavelength);
    params.detailAmplitude   = tuning.detailAmplitude;
    params.detailRidgeWeight = glm::clamp(tuning.detailRidgeWeight, 0.0f, 1.0f);
    params.detailSharpness   = std::max(tuning.detailSharpness, 0.25f);

    params.bandFrequency    = std::max(tuning.bandCount, 0.0f);
    params.bandAmplitude    = tuning.bandAmplitude;
    params.bandSharpness    = std::max(tuning.bandSharpness, 0.1f);
    params.baseHeightOffset = tuning.baseHeightOffset;

    params.cavityFrequency = wavelengthToFrequency(tuning.caveWavelength);
    params.cavityAmplitude = tuning.caveAmplitude;
    params.cavityThreshold = glm::clamp(tuning.caveThreshold, 0.0f, 1.0f);
    params.cavityContrast  = std::max(tuning.caveContrast, 0.1f);

    params.moistureFrequency   = wavelengthToFrequency(tuning.moistureWavelength);
    params.moistureOctaves     = tuning.moistureOctaves;
    params.moisturePersistence = glm::clamp(tuning.moisturePersistence, 0.1f, 0.95f);
    return params;
}

NoiseParams NoiseParams::disabled() {
    NoiseParams n{};
    n.macroFrequency = 0.0f;
    n.macroAmplitude = 0.0f;
    n.macroRidgeWeight = 0.0f;
    n.macroSharpness = 1.0f;

    n.detailFrequency = 0.0f;
    n.detailAmplitude = 0.0f;
    n.detailRidgeWeight = 0.0f;
    n.detailSharpness = 1.0f;

    n.bandFrequency = 0.0f;
    n.bandAmplitude = 0.0f;
    n.bandSharpness = 1.0f;
    n.baseHeightOffset = 0.0f;

    n.cavityFrequency = 0.0f;
    n.cavityAmplitude = 0.0f;
    n.cavityThreshold = 1.0f;
    n.cavityContrast = 1.0f;

    n.moistureFrequency = 0.0f;
    n.moistureOctaves = 0;
    n.moisturePersistence = 0.0f;
    return n;
}

glm::vec3 ApplyDomainWarp(const glm::vec3& surfacePoint,
                          const NoiseParams& noise,
                          std::uint32_t seed) {
    const float freq = noise.detailFrequency * 0.5f;
    const float amp = noise.detailAmplitude * 0.25f;
    if (freq <= 0.0f || amp <= 0.0f) {
        return surfacePoint;
    }
    const glm::vec3 domain = surfacePoint * freq;
    glm::vec3 warp(
        signedValueNoise(domain + glm::vec3(31.7f, 17.3f, 13.1f), seed + 233u),
        signedValueNoise(domain + glm::vec3(11.1f, 53.2f, 27.8f), seed + 389u),
        signedValueNoise(domain + glm::vec3(91.7f, 45.3f, 67.1f), seed + 521u));
    return surfacePoint + warp * amp;
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
