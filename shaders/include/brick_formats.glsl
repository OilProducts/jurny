// GPU-side mirror of BrickHeader used in SSBO buffers.
// Matches the 32-byte C++ layout exactly by packing two uint16_t (flags, paletteCount)
// into a single uint (flagsAndCount), followed by tsdfOffset.
struct BrickHeader {
  int  bx;
  int  by;
  int  bz;
  uint occOffset;       // byte offset into occupancy buffer
  uint matIdxOffset;    // unused in M2
  uint paletteOffset;   // unused in M2
  uint flagsAndCount;   // low 16 bits: flags, high 16 bits: paletteCount
  uint tsdfOffset;      // unused
};
