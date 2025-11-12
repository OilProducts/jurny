#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "math/Spherical.h"
#include "world/BrickFormats.h"
#include "world/BrickStore.h"

namespace {

struct SampleDelta {
    int vx, vy, vz;
    float tsdfCenter;
    float analyticCenter;
    float error;
    bool occupied;
};

void printUsage(const char* exe) {
    std::cerr << "Usage: " << exe << " <bx> <by> <bz> [radius]" << std::endl;
}

math::NoiseTuning buildNoiseTuning() {
    math::NoiseTuning tuning{};
    tuning.continentsPerCircumference = 3.2f;
    tuning.continentAmplitude = 110.0f;
    tuning.continentOctaves = 5;
    tuning.detailWavelength = 140.0f;
    tuning.detailAmplitude = 6.0f;
    tuning.detailOctaves = 2;
    tuning.detailWarpMultiplier = 1.3f;
    tuning.baseHeightOffset = 14.0f;
    tuning.warpWavelength = 240.0f;
    tuning.warpAmplitude = 24.0f;
    tuning.slopeSampleDistance = 100.0f;
    tuning.caveWavelength = 36.0f;
    tuning.caveAmplitude = 5.0f;
    tuning.caveThreshold = 0.35f;
    tuning.moistureWavelength = 80.0f;
    tuning.moistureOctaves = 4;
    return tuning;
}

int divFloor(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if ((r != 0) && ((r < 0) != (b < 0))) --q;
    return q;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    const int targetBx = std::atoi(argv[1]);
    const int targetBy = std::atoi(argv[2]);
    const int targetBz = std::atoi(argv[3]);
    const int radius = (argc >= 5) ? std::max(0, std::atoi(argv[4])) : 0;

    const math::PlanetParams planet{100.0, 120.0, 100.0, 160.0};
    const math::NoiseTuning tuning = buildNoiseTuning();
    const math::NoiseParams noise = math::BuildNoiseParams(tuning, planet);
    constexpr uint32_t worldSeed = 1337u;

    world::BrickStore store;
    store.configure(planet, /*voxelSize*/0.5f, VOXEL_BRICK_SIZE, noise, worldSeed, nullptr);

    std::vector<glm::ivec3> requested;
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                requested.emplace_back(targetBx + dx, targetBy + dy, targetBz + dz);
            }
        }
    }
    if (requested.empty()) {
        requested.emplace_back(targetBx, targetBy, targetBz);
    }

    auto cpu = store.buildCpuWorld(requested);
    if (cpu.headers.empty()) {
        std::cerr << "BrickStore produced no bricks (will fall back to analytic field only)." << std::endl;
    }

    const auto findHeader = [&](const glm::ivec3& bc) -> std::optional<size_t> {
        for (size_t i = 0; i < cpu.headers.size(); ++i) {
            const auto& h = cpu.headers[i];
            if (h.bx == bc.x && h.by == bc.y && h.bz == bc.z) {
                return i;
            }
        }
        return std::nullopt;
    };

    const auto targetIndex = findHeader({targetBx, targetBy, targetBz});
    world::BrickHeader header{};
    if (targetIndex) {
        header = cpu.headers[*targetIndex];
    } else {
        header.bx = targetBx;
        header.by = targetBy;
        header.bz = targetBz;
        header.occOffset = 0;
        header.tsdfOffset = world::kInvalidOffset;
    }
    const int brickDim = store.brickDim();
    const size_t voxelsPerBrick = static_cast<size_t>(brickDim) * brickDim * brickDim;
    const uint64_t* occBase = nullptr;
    if (targetIndex) {
        occBase = cpu.occWords.data() + (header.occOffset / sizeof(uint64_t));
    }
    const float* fieldBase = nullptr;
    if (targetIndex && header.tsdfOffset != world::kInvalidOffset && !cpu.fieldSamples.empty()) {
        fieldBase = cpu.fieldSamples.data() + (header.tsdfOffset / sizeof(float));
    }

    const glm::vec3 brickOrigin = glm::vec3(header.bx, header.by, header.bz) * store.brickSize();
    const float voxelSize = store.voxelSize();

    auto isOccupied = [&](int vx, int vy, int vz) -> bool {
        if (!occBase) return false;
        const uint32_t linear = static_cast<uint32_t>(vx + vy * brickDim + vz * brickDim * brickDim);
        const uint32_t word = linear >> 6u;
        const uint32_t bit = linear & 63u;
        return ((occBase[word] >> bit) & 1ull) != 0ull;
    };

    const int samplesPerAxis = world::FieldSamplesPerAxis(brickDim);
    const int fieldResolution = world::kFieldResolution;
    const int fieldApron = world::kFieldApron;
    const int sampleApronOffset = fieldApron * fieldResolution;

    auto sampleGrid = [&](int ix, int iy, int iz) -> float {
        ix = std::clamp(ix, 0, samplesPerAxis - 1);
        iy = std::clamp(iy, 0, samplesPerAxis - 1);
        iz = std::clamp(iz, 0, samplesPerAxis - 1);
        const size_t idx = (static_cast<size_t>(iz) * samplesPerAxis + static_cast<size_t>(iy)) * samplesPerAxis + static_cast<size_t>(ix);
        return fieldBase ? fieldBase[idx] : std::numeric_limits<float>::quiet_NaN();
    };

    auto sampleTrilinear = [&](float x, float y, float z) -> float {
        x = std::clamp(x, 0.0f, static_cast<float>(samplesPerAxis - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(samplesPerAxis - 1));
        z = std::clamp(z, 0.0f, static_cast<float>(samplesPerAxis - 1));
        int ix = static_cast<int>(std::floor(x));
        int iy = static_cast<int>(std::floor(y));
        int iz = static_cast<int>(std::floor(z));
        int ix1 = std::min(ix + 1, samplesPerAxis - 1);
        int iy1 = std::min(iy + 1, samplesPerAxis - 1);
        int iz1 = std::min(iz + 1, samplesPerAxis - 1);
        const float tx = x - static_cast<float>(ix);
        const float ty = y - static_cast<float>(iy);
        const float tz = z - static_cast<float>(iz);
        const float c000 = sampleGrid(ix, iy, iz);
        const float c100 = sampleGrid(ix1, iy, iz);
        const float c010 = sampleGrid(ix, iy1, iz);
        const float c110 = sampleGrid(ix1, iy1, iz);
        const float c001 = sampleGrid(ix, iy, iz1);
        const float c101 = sampleGrid(ix1, iy, iz1);
        const float c011 = sampleGrid(ix, iy1, iz1);
        const float c111 = sampleGrid(ix1, iy1, iz1);
        const float c00 = std::lerp(c000, c100, tx);
        const float c10 = std::lerp(c010, c110, tx);
        const float c01 = std::lerp(c001, c101, tx);
        const float c11 = std::lerp(c011, c111, tx);
        const float c0 = std::lerp(c00, c10, ty);
        const float c1 = std::lerp(c01, c11, ty);
        return std::lerp(c0, c1, tz);
    };

    auto sampleCenterField = [&](int vx, int vy, int vz) -> float {
        if (!fieldBase) return std::numeric_limits<float>::quiet_NaN();
        const float samplesPerVoxel = static_cast<float>(std::max(1, fieldResolution));
        const float relX = static_cast<float>(sampleApronOffset) + (static_cast<float>(vx) + 0.5f) * samplesPerVoxel;
        const float relY = static_cast<float>(sampleApronOffset) + (static_cast<float>(vy) + 0.5f) * samplesPerVoxel;
        const float relZ = static_cast<float>(sampleApronOffset) + (static_cast<float>(vz) + 0.5f) * samplesPerVoxel;
        return sampleTrilinear(relX, relY, relZ);
    };

    std::vector<SampleDelta> samples;
    samples.reserve(voxelsPerBrick);

    float minTsdf = std::numeric_limits<float>::infinity();
    float maxTsdf = -std::numeric_limits<float>::infinity();
    float minAnalytic = std::numeric_limits<float>::infinity();
    float maxAnalytic = -std::numeric_limits<float>::infinity();
    float minError = std::numeric_limits<float>::infinity();
    float maxError = -std::numeric_limits<float>::infinity();
    size_t occupiedCount = 0;

    for (int vz = 0; vz < brickDim; ++vz) {
        for (int vy = 0; vy < brickDim; ++vy) {
            for (int vx = 0; vx < brickDim; ++vx) {
                const bool occ = isOccupied(vx, vy, vz);
                if (occ) ++occupiedCount;

                const float tsdfCenter = sampleCenterField(vx, vy, vz);
                const glm::vec3 centerWorld = brickOrigin + (glm::vec3(vx, vy, vz) + glm::vec3(0.5f)) * voxelSize;
                const float analytic = math::F_crust(centerWorld, planet, noise, worldSeed);
                const float error = tsdfCenter - analytic;

                minTsdf = std::min(minTsdf, tsdfCenter);
                maxTsdf = std::max(maxTsdf, tsdfCenter);
                minAnalytic = std::min(minAnalytic, analytic);
                maxAnalytic = std::max(maxAnalytic, analytic);
                minError = std::min(minError, error);
                maxError = std::max(maxError, error);

                samples.push_back({vx, vy, vz, tsdfCenter, analytic, error, occ});
            }
        }
    }

    std::vector<SampleDelta> worst = samples;
    std::sort(worst.begin(), worst.end(), [](const SampleDelta& a, const SampleDelta& b) {
        return std::abs(a.error) > std::abs(b.error);
    });
    if (worst.size() > 10) worst.resize(10);

    const int macroDim = 8;
    const glm::ivec3 regionCoord{
        divFloor(header.bx, macroDim),
        divFloor(header.by, macroDim),
        divFloor(header.bz, macroDim)};

    std::cout << "Brick (" << header.bx << ", " << header.by << ", " << header.bz << ")\n";
    std::cout << "  Region: (" << regionCoord.x << ", " << regionCoord.y << ", " << regionCoord.z << ")\n";
    std::cout << "  World origin: (" << brickOrigin.x << ", " << brickOrigin.y << ", " << brickOrigin.z << ") meters\n";
    if (!targetIndex) {
        std::cout << "  NOTE: brick was outside the streamed crust (BrickStore skipped it); showing analytic field only.\n";
    }
    std::cout << "  Occupied voxels: " << occupiedCount << " / " << voxelsPerBrick
              << " (" << (100.0 * static_cast<double>(occupiedCount) / static_cast<double>(voxelsPerBrick)) << "%)\n";
    if (fieldBase) {
        std::cout << "  TSDF center range: [" << minTsdf << ", " << maxTsdf << "] meters\n";
    } else {
        std::cout << "  TSDF center range: n/a (no cached field)\n";
    }
    std::cout << "  Analytic F range: [" << minAnalytic << ", " << maxAnalytic << "] meters\n";
    if (fieldBase) {
        std::cout << "  TSDF - Analytic error range: [" << minError << ", " << maxError << "] meters\n";
    }

    if (fieldBase) {
        std::cout << "\nTop |error| voxels (vx, vy, vz):\n";
        for (const auto& s : worst) {
            std::cout << "  (" << s.vx << ", " << s.vy << ", " << s.vz << ")"
                      << " occ=" << (s.occupied ? "Y" : "N")
                      << " tsdf=" << s.tsdfCenter
                      << " analytic=" << s.analyticCenter
                      << " error=" << s.error << '\n';
        }
    }

    std::cout << "\nPer-plane summary (vz slices):\n";
    for (int vz = 0; vz < brickDim; ++vz) {
        double sliceError = 0.0;
        double sliceAbs = 0.0;
        size_t sliceCount = 0;
        for (const auto& s : samples) {
            if (s.vz != vz) continue;
            sliceError += s.error;
            sliceAbs += std::abs(s.error);
            ++sliceCount;
        }
        if (sliceCount == 0) continue;
        std::cout << "  vz=" << vz
                  << " meanError=" << (sliceError / static_cast<double>(sliceCount))
                  << " meanAbsError=" << (sliceAbs / static_cast<double>(sliceCount))
                  << '\n';
    }

    auto sampleLookup = [&](int vx, int vy, int vz) -> const SampleDelta& {
        const size_t idx = static_cast<size_t>(vx) + static_cast<size_t>(vy) * brickDim +
                           static_cast<size_t>(vz) * brickDim * brickDim;
        return samples[idx];
    };

    std::cout << "\nAnalytic center field per slice (meters):\n";
    for (int vz = 0; vz < brickDim; ++vz) {
        std::cout << "  vz=" << vz << '\n';
        for (int vy = 0; vy < brickDim; ++vy) {
            std::cout << "    vy=" << vy << ":";
            for (int vx = 0; vx < brickDim; ++vx) {
                const auto& s = sampleLookup(vx, vy, vz);
                std::cout << ' ' << s.analyticCenter;
            }
            std::cout << '\n';
        }
    }

    if (fieldBase) {
        std::cout << "\nTSDF center field per slice (meters):\n";
        for (int vz = 0; vz < brickDim; ++vz) {
            std::cout << "  vz=" << vz << '\n';
            for (int vy = 0; vy < brickDim; ++vy) {
                std::cout << "    vy=" << vy << ":";
                for (int vx = 0; vx < brickDim; ++vx) {
                    const auto& s = sampleLookup(vx, vy, vz);
                    std::cout << ' ' << s.tsdfCenter;
                }
                std::cout << '\n';
            }
        }
    }

    return EXIT_SUCCESS;
}
