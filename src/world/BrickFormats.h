#pragma once

#include <cstdint>

// BrickFormats — packed GPU layouts and flags.
namespace world {
constexpr uint32_t kInvalidOffset = 0xFFFFFFFFu;
constexpr int      kFieldApron    = 1;

enum BrickFlags : uint16_t {
    kBrickUses4Bit = 1u << 0
};

struct BrickHeader {
    int32_t bx{}, by{}, bz{};
    uint32_t occOffset{};
    uint32_t matIdxOffset{};
    uint32_t paletteOffset{};
    uint16_t flags{};
    uint16_t paletteCount{};
    uint32_t tsdfOffset{}; // 0xFFFFFFFF if none
};

struct MaterialGpu {
    float baseColor[3];
    float roughness;
    float emission;
    float metalness;
    float pad[2];
};

}
