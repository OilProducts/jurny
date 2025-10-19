// GPU-side mirror of BrickHeader used in SSBO buffers.
// Matches the 32-byte C++ layout exactly by packing two uint16_t (flags, paletteCount)
// into a single uint (flagsAndCount), followed by tsdfOffset.
struct BrickHeader {
  int  bx;
  int  by;
  int  bz;
  uint occOffset;       // byte offset into occupancy buffer
  uint matIdxOffset;    // byte offset into material index buffer
  uint paletteOffset;   // unused (reserved for compact palettes)
  uint flagsAndCount;   // low 16 bits: flags, high 16 bits: paletteCount
  uint tsdfOffset;      // unused (reserved for TSDF tiles)
};

uint brickFlags(in BrickHeader h) {
  return h.flagsAndCount & 0xFFFFu;
}

const uint BRICK_FLAG_USES4BIT = 1u << 0;
const uint BRICK_INVALID_OFFSET = 0xFFFFFFFFu;

uint brickPaletteCount(in BrickHeader h) {
  return h.flagsAndCount >> 16;
}

struct MaterialGpu {
  vec4 baseColorRoughness;   // rgb + roughness
  vec4 emissionMetalnessPad; // emission, metalness, padding
};
