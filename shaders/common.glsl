// common.glsl — shared types, bindings, RNG helpers.
// Mirrors C++ GlobalsUBO layout; included by compute passes.

struct GlobalsUBO_t {
  // Camera (local space, origin rebased)
  mat4 currView;
  mat4 currProj;
  mat4 currViewInv;
  mat4 currProjInv;
  mat4 prevView;
  mat4 prevProj;

  // Origin used to rebase world->camera transforms (world space, meters)
  vec3 renderOrigin; float _padOrigin;

  // Origin delta (prev→curr) in local space
  vec3 originDeltaPrevToCurr; float _pad0;

  // Planet + render params
  float voxelSize;  float brickSize;  float Rin;       float Rout;
  float Rsea;       float planetRadius; float exposure; float _padExposure;

  // Frame / image info
  uint frameIdx; uint maxBounces; uint width; uint height;

  uint raysPerPixel; uint flags; uint worldHashCapacity; uint worldBrickCount;
  // World hash info for traversal
  uint macroHashCapacity;
  uint macroDimBricks;
  float macroSize;
  uint historyValid;
  float noiseContinentFreq;
  float noiseContinentAmp;
  float noiseDetailFreq;
  float noiseDetailAmp;
  float noiseWarpFreq;
  float noiseWarpAmp;
  float noiseCaveFreq;
  float noiseCaveAmp;
  float noiseCaveThreshold;
  float noiseMinHeight;
  float noiseMaxHeight;
  float noiseDetailWarp;
  float noiseSlopeSampleDist;
  float noiseBaseHeightOffset;
  float noisePad2;
  uint noiseSeed;
  uint noiseContinentOctaves;
  uint noiseDetailOctaves;
  uint noiseCaveOctaves;
};

#ifndef GLOBALS_UBO_DEFINED
#define GLOBALS_UBO_DEFINED 1
layout(set = 0, binding = 0) uniform GlobalsUBO { GlobalsUBO_t g; };
#endif

// layout(set=0, binding=0) uniform GlobalsUBO { GlobalsUBO_t g; };
// Note: bind this only in passes that need it; first_pixels.comp does not include this file.

// Simple PCG-like hash for RNG seed per pixel
uint pcg_hash(uint x) {
  x ^= x >> 17; x *= 0xed5ad4bbu;
  x ^= x >> 11; x *= 0xac4c1b51u;
  x ^= x >> 15; x *= 0x31848babu;
  x ^= x >> 14; return x;
}
