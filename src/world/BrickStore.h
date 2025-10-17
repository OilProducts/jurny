#pragma once

#include <vector>
#include <cstdint>
#include "BrickFormats.h"
#include "math/Spherical.h"

// BrickStore — CPU brick generation & packing for a test slab near +X.
// GPU upload is handled by the renderer for M2.
namespace world {

struct CpuWorld {
    std::vector<BrickHeader> headers;
    std::vector<uint64_t>    occWords;   // 8x uint64 per brick
    std::vector<uint64_t>    hashKeys;   // packed (bx,by,bz)
    std::vector<uint32_t>    hashVals;   // index into headers
    uint32_t                 hashCapacity = 0; // power-of-two
    // Macro mask (non-empty macro tiles)
    std::vector<uint64_t>    macroKeys;  // packed macro coords
    std::vector<uint32_t>    macroVals;  // 1 = non-empty tile
    uint32_t                 macroCapacity = 0;
    uint32_t                 macroDimBricks = 8; // default grouping: 8x8x8 bricks per macro tile
};

class BrickStore {
public:
    void clear();

    // Generate a small curved "terrain" slab that approximates the spherical crust
    // centered around +X axis. Returns number of bricks.
    size_t generateTestSlab(const math::PlanetParams& P,
                            float voxelSize,
                            int brickDim,
                            float yExtentMeters,
                            float zExtentMeters,
                            float radialHalfThicknessMeters);

    // Generate a spherical cap centered around +X axis: for each (by,bz) within tangential extents,
    // place bricks around the analytic surface x = sqrt(R^2 - y^2 - z^2) with given radial half thickness.
    size_t generateSphericalCap(const math::PlanetParams& P,
                                float voxelSize,
                                int brickDim,
                                float yExtentMeters,
                                float zExtentMeters,
                                float radialHalfThicknessMeters);

    const CpuWorld& cpu() const { return cpu_; }

private:
    static uint64_t packKey(int bx, int by, int bz);
    void buildHash();
    void buildMacroHash(uint32_t macroDimBricks);

private:
    int brickDim_ = 8;
    float brickSize_ = 4.0f;
    float voxelSize_ = 0.5f;
    math::PlanetParams params_{};
    CpuWorld cpu_{};
};

}
