// common.glsl — shared types, bindings, RNG helpers.
// Mirrors C++ GlobalsUBO layout; included by compute passes.

struct GlobalsUBO_t {
  // Camera (local space, origin rebased)
  mat4 currView;
  mat4 currProj;
  mat4 prevView;
  mat4 prevProj;

  // Origin delta (prev→curr) in local space
  vec3 originDeltaPrevToCurr; float _pad0;

  // Planet + render params
  float voxelSize;  float brickSize;  float Rin;       float Rout;
  float Rsea;       float exposure;   uint  frameIdx;  uint  maxBounces;

  // Dimensions
  uint width; uint height; uint raysPerPixel; uint flags;
  // World hash info for traversal
  uint worldHashCapacity; uint worldBrickCount; uint _padA; uint _padB;
  // Macro skipping
  uint macroHashCapacity; uint macroDimBricks; float macroSize; float _padC;
};

// layout(set=0, binding=0) uniform GlobalsUBO { GlobalsUBO_t g; };
// Note: bind this only in passes that need it; first_pixels.comp does not include this file.

// Simple PCG-like hash for RNG seed per pixel
uint pcg_hash(uint x) {
  x ^= x >> 17; x *= 0xed5ad4bbu;
  x ^= x >> 11; x *= 0xac4c1b51u;
  x ^= x >> 15; x *= 0x31848babu;
  x ^= x >> 14; return x;
}
