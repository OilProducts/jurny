Spherical World Math — Implementation Outline (Part 4)

Purpose
- Define the math that makes the world “planetary” while keeping a plain Cartesian, axis‑aligned voxel grid.
- Specify robust formulas, data flow (CPU doubles → GPU floats via origin rebase), and APIs you’ll implement.
- Provide concrete tolerances, defaults, and tests so you can type with confidence.

Invariants & Scope
- World is R^3 with the planet as a region selected by a continuous signed field F(p).
- Bricks are axis‑aligned AABBs in world space. No curved bricks or projections.
- CPU uses geocentric doubles; GPU uses camera‑relative floats based on a per‑frame origin rebase.
- Gravity is radial; “up” is the normalized position vector at a point.
- Rays are clamped to a spherical shell before voxel DDA to avoid wasted traversal.

Key Parameters (PlanetParams)
- R: base planet radius (meters).
- T: effective crust thickness below terrain (meters).
- R_sea: sea level radius (meters).
- H_max: max positive terrain height above R (meters) used for shell bounds.
- Optional: AtmosphereTop (meters above R) for sky cutoff.

Coordinate Spaces
1) CPU Global (geocentric, double)
   - All persistent world transforms/positions in double precision.
   - The camera world position `camWorld` is the rebase origin each frame.

2) GPU Local (camera‑relative, float)
   - `local = float3(world - originWorld)`
   - All render math uses local floats; precision remains stable anywhere on the globe.

Per‑Frame Origin Rebase
- worldOriginCurr (double) = camera world position.
- worldOriginPrev (double) = previous frame’s origin.
- On CPU, when preparing draw constants:
  - currView, currProj built in local space based on `worldOriginCurr`.
  - prevView, prevProj built in local space based on `worldOriginPrev`.
  - originDeltaPrevToCurr = (worldOriginCurr - worldOriginPrev) cast to float3.
- On GPU, positions from this frame are in the “curr local” frame; when reprojecting to previous frame, subtract `originDeltaPrevToCurr` before applying prev matrices (see Motion Vectors below).

Local Tangent Frame & “Up”
- up(p) = normalize(p_local)   // p_local in current local space
- east = normalize(cross((0,0,1), up)); if |east|^2 < eps → east = (1,0,0) (polar fallback)
- north = cross(up, east)
- Use ENU basis for: brushes, character controllers, gizmos, and local AABB alignment when desired.

Gravity
- Constant magnitude near surface: g(p) = -g0 * up(p)
- Or inverse‑square (optional): g(p) = -g0 * (R^2 / |p_world|^2) * up_world(p)
- For small simulation AABBs, treat g constant within the region; refresh if the AABB moves far.

Planet Signed Fields (continuous; negative = solid)
Terminology
- p: position (float3) in local space for GPU; (double3) in world space on CPU.
- u = normalize(p_world) (unit direction on sphere).

Base terrain height (meters)
- H(u) = domain‑warped noise sampled on the unit sphere u (seam‑free). Amplitude in meters.

Crust altitude above terrain
- f_crust(p_world) = |p_world| − (R + H(u))

Finite crust thickness + caves (optional)
- C(p_world) = band‑passed 3D noise in world space (0..1). Let caveAmp be thickness budget in meters.
- F(p_world) = max(f_crust(p_world), −(T − C(p_world)*caveAmp))
  - Negative values → solid; positive → air.

Normals (for shading & collision)
- n(p) = normalize(∇F(p)) computed via central differences in meters.
- Recommended epsilon for gradient sampling: eps_n = 0.5 * voxelSize (clamped to [0.25, 1.5] * voxelSize).

Ray Clamp to Spherical Shell
Goal
- Avoid stepping the voxel DDA outside where content can exist.

Radii
- Rin = R − T − marginBelow
- Rout = R + H_max + AtmosphereTop

Intersect sphere of radius R (unit‑length ray direction d assumed)
  b = dot(o, d)
  c = dot(o, o) − R^2
  disc = b^2 − c
  if disc < 0 → no hit
  t0 = −b − sqrt(disc); t1 = −b + sqrt(disc)

Shell clamp (outer − inner)
1) Intersect outer sphere (Rout) → [o0,o1]. If no hit → miss.
2) Intersect inner sphere (Rin) → [i0,i1] if it exists.
3) Shell intervals along the ray are: (-∞, i0] ∩ [o0,o1] and [i1, +∞) ∩ [o0,o1].
4) Choose the nearest interval with positive length and overlapping [0,+∞).
5) Set ray.tmin = max(ray.tmin, tEnter); ray.tmax = min(ray.tmax, tExit).

Practical notes
- Normalize `d` before using the quadratic (or account for |d| in the math).
- Clamp tmin to a small epsilon (e.g., 1e‑4 m) to avoid self‑hits.
- When starting inside the shell, `tEnter` can be 0.

AABB vs Spherical Shell (for streaming culling)
- AABB b with center c and half‑extents h (world meters).
- min radius to AABB: let q = max(|c| − h, 0); r_min = |q|
- max radius to AABB: r_max = | |c| + h |
- Overlap test: (r_min ≤ Rout) && (r_max ≥ Rin)

Brick Addressing (still axis‑aligned)
- voxelSize (meters), B = 8, brickSize S = B * voxelSize.
- brickCoord(p) = floor(p / S)  // integer triple (bx,by,bz)
- brickOrigin(bc) = bc * S
- voxelCoord(p) = clamp(floor(p / voxelSize) − B * floor(p / S), 0..B‑1)
- Hash key for lookup: pack (bx,by,bz) into 64‑bit; e.g., bias 21‑bit signed per axis.

Surface Hit Refinement to F(p)=0 (removes “corner‑up” artifacts)
Context
- Brick micro‑DDA yields the first solid cell segment [t0, t1] along the ray (in meters, local frame).

Refinement (bisection, robust)
1) Evaluate F(o + t0 d) and F(o + t1 d). If signs are equal, slightly expand by a tiny bias (e.g., ±0.1*voxelSize) or accept t0 as fallback.
2) Do 6–8 bisection steps to find t* where F≈0.
3) Hit point p* = o + t* d.
4) Normal n = normalize(∇F(p*)) using central differences (eps_n above).

TSDF Override (edited/dynamic bricks)
- If a brick has a TSDF tile, use it for refinement and normals instead of analytic F. TSDF stores signed meters (or scaled int16_t) truncated to a small band.
- Update TSDF when edits occur via jump‑flood or fast sweeping; include a 1‑voxel halo to keep gradients continuous across bricks.

Motion Vectors & Temporal Reprojection (with rebasing)
Definitions
- P_world_curr: world position (double) of the current pixel’s hit.
- originPrev, originCurr (double): previous and current rebasing origins.
- P_local_curr = (P_world_curr − originCurr) cast to float3.
- P_local_prev = (P_world_curr − originPrev) cast to float3.

Compute motion vector (NDC)
1) prevClip = prevProj * prevView * vec4(P_local_prev, 1)
2) currClip = currProj * currView * vec4(P_local_curr, 1)
3) prevNDC = prevClip.xyz / prevClip.w
4) currNDC = currClip.xyz / currClip.w
5) velocity = 0.5 * (currNDC.xy − prevNDC.xy)   // map from [-1,1] to [-0.5,0.5]

Notes
- This remains valid regardless of how far the camera moves because the world‑to‑local difference is accounted for by using different origins per frame.
- Provide material ID, normal, and plane depth for history validation/clamping in the denoiser.

Streaming Policy on a Sphere
Windows
- Radius window: [Rin, Rout] (as above).
- Angular window: accept regions/bricks whose center direction u_b satisfies acos(dot(u_b, u_cam)) ≤ θ_max.

Priority score per region/brick
- score = w0/(1 + angDist) + w1/(1 + |radius − R_sea|) + w2*recentlyVisible + w3*simActive − w4*age
- Promote sim‑active and recently visible bricks; evict by low score + age when over budget.

Numerical Tolerances & Robustness
- t epsilon (ray start): t_min = max(userNear, 1e‑4 * voxelSize).
- Step bias when leaving a cell: add 1e‑4 * voxelSize in ray parameter to avoid re‑hitting the same face.
- Gradient epsilon: eps_n = 0.5 * voxelSize (clamp as noted).
- Bisection steps: 6–8 sufficient for sub‑voxel precision; stop if interval width < 0.2 * voxelSize.
- Hash table load factor: ≤ 0.5 to keep probe lengths short and coherent.
- Planet radii: compute shell intersections in float on GPU, but prefer double on CPU for worldgen and culling.

Recommended Defaults (dev scale “small moon”)
- R = 10,000 m
- T = 60 m (crust thickness budget for caves)
- R_sea = R (sea at base radius)
- H_max = 1,000 m (mountains)
- AtmosphereTop = 5,000 m
- voxelSize = 0.25 m; brick B=8 → brickSize S=2 m

Core APIs (headers you’ll implement)
// C++ (src/math/Spherical.h)
- struct PlanetParams { double R, T, sea, Hmax, atmosphereTop; };
- struct NoiseParams { … };  // shared with WorldGen + renderer (continents, detail, warp, caves, moisture)
- inline constexpr int kNoiseCaveOctaves = 4;
- struct CrustSample { float field; float height; };
- CrustSample SampleCrust(glm::vec3 p_local, const PlanetParams&, const NoiseParams&, uint32_t seed);
- float  F_crust(glm::vec3 p_local, const PlanetParams&, const NoiseParams&, uint32_t seed);
- glm::vec3 gradF(glm::vec3 p_local, const PlanetParams&, const NoiseParams&, uint32_t seed, float epsMeters);
- bool IntersectSphereShell(glm::vec3 o_local, glm::vec3 d_local,
                            float Rin, float Rout,
                            float& tEnter, float& tExit);
- void ENU(const glm::vec3& p_local, glm::vec3& east, glm::vec3& north, glm::vec3& up);
- glm::vec3 ApplyDomainWarp(glm::vec3 unitDir, const NoiseParams&, uint32_t seed);
- float FractalBrownianMotion(glm::vec3 p, float baseFrequency, int octaves, float persistence, uint32_t seed);

// GLSL (shaders/common.glsl)
- bool intersectSphere(vec3 o, vec3 d, float R, out float t0, out float t1);
- bool intersectSphereShell(vec3 o, vec3 d, float Rin, float Rout,
                            out float tEnter, out float tExit);
- vec3 gradF(vec3 p, float epsMeters);
- float F_crust(vec3 p);

Data in GlobalsUBO (required by math)
- camPos (local), currView, currProj, prevView, prevProj
- originDeltaPrevToCurr (local float3)
- voxelSize, brickSize
- Rin, Rout, R_sea
- frameIdx, maxBounces (not math, but grouped here)
- width, height, raysPerPixel, flags

Integration Points (where this math is used)
- generate_rays.comp: build primary rays in local space, set initial t range.
- traverse_bricks.comp: shell clamp → DDA stepping → micro‑DDA per brick → bracket [t0,t1] → refine to F=0 → normals.
- shade.comp: use normals, material from palette; compute motion vectors using prev/curr transforms and origin delta.
- denoise_atrous.comp: consumes velocity, normal, albedo, variance; history validation uses material ID and normal agreement.
- world streaming (CPU): AABB vs shell overlap for region selection; angular scoring; double precision positions but store local AABBs for uploads.

Edge Cases & Handling
- Starting inside inner sphere: Intervals from [i1, o1] only; clamp tEnter=0.
- Inside solid (F<0) at camera: nudge ray start by small epsilon along +d.
- Grazing hits (tangent rays): discriminant ~ 0; treat as miss unless bracket has width > small threshold.
- Poles (east vector degeneracy): fallback east=(1,0,0) when |cross(z,up)| is tiny.
- Very high altitude: rely on shell clamp; if outside Rout → early miss.

Testing Plan (unit tests)
- Sphere intersection: analytical cases (origin outside, inside, tangent) with tiny eps tolerances.
- Shell clamp: exhaustive discrete directions against brute‑force marching baseline.
- AABB vs shell: crafted boxes that just touch inner/outer radii.
- Gradient check: directional derivative test of F using finite differences.
- Reprojection: property test that static point on ground yields near‑zero velocity when camera only rotates; non‑zero when translating without rebase; zero when translating with rebase.

Telemetry to Expose (for verification)
- % rays culled by shell clamp
- Avg brick probes per lookup; probe length histogram
- Micro‑DDA steps per hit/miss
- Avg/95th refinement iterations; % of TSDF vs analytic used
- Motion vector magnitude heatmap; history rejection rate

Reference Snippets (ready to paste)

// GLSL: sphere + shell intersection
bool intersectSphere(vec3 o, vec3 d, float R, out float t0, out float t1) {
  float b = dot(o, d);
  float c = dot(o, o) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  t0 = -b - s; t1 = -b + s;
  return t1 > 0.0;
}

bool intersectSphereShell(vec3 o, vec3 d, float Rin, float Rout, out float tEnter, out float tExit) {
  float o0, o1; if (!intersectSphere(o, d, Rout, o0, o1)) return false;
  float i0, i1; bool hitIn = intersectSphere(o, d, Rin, i0, i1);
  float a0 = max(o0, 0.0), a1 = hitIn ? min(o1, i0) : o1;
  float b0 = hitIn ? max(i1, 0.0) : 1e30, b1 = o1;
  if (a1 > a0) { tEnter = a0; tExit = a1; return true; }
  if (b1 > b0) { tEnter = b0; tExit = b1; return true; }
  return false;
}

// GLSL: robust bisection refine to F=0
float refineIso(vec3 o, vec3 d, float t0, float t1) {
  float f0 = F_crust(o + t0*d);
  float f1 = F_crust(o + t1*d);
  if (f0 * f1 > 0.0) return t0; // fallback (no bracket)
  for (int i = 0; i < 7; ++i) {
    float tm = 0.5*(t0 + t1);
    float fm = F_crust(o + tm*d);
    bool same = (fm * f0) > 0.0;
    t0 = same ? tm : t0; f0 = same ? fm : f0;
    t1 = same ? t1 : tm;
  }
  return 0.5*(t0 + t1);
}

// C++: AABB vs shell overlap
struct AABB { glm::dvec3 c; glm::dvec3 h; };
inline double minRadiusToAABB0(const AABB& b) {
  glm::dvec3 q = glm::max(glm::abs(b.c) - b.h, glm::dvec3(0));
  return glm::length(q);
}
inline double maxRadiusToAABB0(const AABB& b) {
  return glm::length(glm::abs(b.c) + b.h);
}
inline bool overlapsShell(const AABB& b, double Rin, double Rout) {
  double dmin = minRadiusToAABB0(b);
  double dmax = maxRadiusToAABB0(b);
  return (dmin <= Rout) && (dmax >= Rin);
}

Why This Produces a Planet Without Curved Bricks
- Curvature arises from the selection field F(p), the radial up/ENU orientation, and the shell clamp. The grid remains a regular Cartesian lattice used for storage and acceleration.
- Visual smoothness comes from refining the hit to the isosurface and computing normals from ∇F (or TSDF) rather than from voxel face normals.
- Streaming uses radius + angle rather than planar rings, ensuring bounded memory and IO around a globe.

Checklist (before shipping M2–M4)
- [ ] `GlobalsUBO` includes Rin/Rout/R_sea, prev/curr matrices, origin delta.
- [ ] Shell clamp in traverse kernel returns early for misses outside shell.
- [ ] Micro‑DDA supplies a valid [t0,t1] bracket for solid cells.
- [ ] Refinement + ∇F normals integrated in shading path.
- [ ] Motion vectors account for rebase; ghosts clamped with variance.
- [ ] CPU streaming uses AABB vs shell + angular scoring.
- [ ] Unit tests for intersections, gradients, and reprojection pass.
