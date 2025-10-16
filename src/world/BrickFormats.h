#pragma once

#include <cstdint>

// BrickFormats — packed GPU layouts and flags.
namespace world {
struct BrickHeader {
    int32_t bx{}, by{}, bz{};
    uint32_t occOffset{};
    uint32_t matIdxOffset{};
    uint32_t paletteOffset{};
    uint16_t flags{};
    uint16_t paletteCount{};
    uint32_t tsdfOffset{}; // 0xFFFFFFFF if none
};
}

