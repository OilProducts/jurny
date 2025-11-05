// spherical.glsl — planet signed field F(p), gradF, and shell/sphere intersection.
// Keep in sync with world/WorldGen.cpp for crustField.

const float PERSISTENCE_CONTINENT = 0.55;
const float PERSISTENCE_DETAIL    = 0.5;
const float PERSISTENCE_CAVE      = 0.5;

float hash13(vec3 p, float seed) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719)) + seed) * 43758.5453);
}

uint pcgHash32(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

uint latticeHash(ivec3 c, uint seed) {
    uvec3 u = uvec3(c);
    uint v = pcgHash32(u.x ^ seed);
    v = pcgHash32(u.y ^ v);
    v = pcgHash32(u.z ^ v);
    return v;
}

float fade1(float t) {
    return t * t * (3.0 - 2.0 * t);
}

vec3 fade3(vec3 v) {
    return vec3(fade1(v.x), fade1(v.y), fade1(v.z));
}

float valueNoise(vec3 p, float seed) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = fade3(f);
    ivec3 cell = ivec3(i);
    uint baseSeed = uint(seed);
    float c000 = hash13(i + vec3(0.0, 0.0, 0.0), float(latticeHash(cell + ivec3(0, 0, 0), baseSeed)));
    float c100 = hash13(i + vec3(1.0, 0.0, 0.0), float(latticeHash(cell + ivec3(1, 0, 0), baseSeed)));
    float c010 = hash13(i + vec3(0.0, 1.0, 0.0), float(latticeHash(cell + ivec3(0, 1, 0), baseSeed)));
    float c110 = hash13(i + vec3(1.0, 1.0, 0.0), float(latticeHash(cell + ivec3(1, 1, 0), baseSeed)));
    float c001 = hash13(i + vec3(0.0, 0.0, 1.0), float(latticeHash(cell + ivec3(0, 0, 1), baseSeed)));
    float c101 = hash13(i + vec3(1.0, 0.0, 1.0), float(latticeHash(cell + ivec3(1, 0, 1), baseSeed)));
    float c011 = hash13(i + vec3(0.0, 1.0, 1.0), float(latticeHash(cell + ivec3(0, 1, 1), baseSeed)));
    float c111 = hash13(i + vec3(1.0, 1.0, 1.0), float(latticeHash(cell + ivec3(1, 1, 1), baseSeed)));

    float nx00 = mix(c000, c100, u.x);
    float nx10 = mix(c010, c110, u.x);
    float nx01 = mix(c001, c101, u.x);
    float nx11 = mix(c011, c111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

float fbm(vec3 p, float baseFrequency, int octaves, float persistence, float seedOffset) {
    float amplitude = 1.0;
    float frequency = baseFrequency;
    float sum = 0.0;
    float norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude * valueNoise(p * frequency, seedOffset + float(i) * 97.0);
        norm += amplitude;
        frequency *= 2.0;
        amplitude *= persistence;
    }
    if (norm <= 0.0) return 0.0;
    float result = sum / norm;
    return result * 2.0 - 1.0;
}

vec3 domainWarp(vec3 p) {
    if (g.noiseWarpAmp <= 0.0 || g.noiseWarpFreq <= 0.0) return p;
    int detailOct = max(int(g.noiseDetailOctaves), 1);
    float fx = fbm(p + vec3(31.7, 17.3, 13.1), g.noiseWarpFreq, detailOct, PERSISTENCE_DETAIL, float(g.noiseSeed) + 233.0);
    float fy = fbm(p + vec3(11.1, 53.2, 27.8), g.noiseWarpFreq, detailOct, PERSISTENCE_DETAIL, float(g.noiseSeed) + 389.0);
    float fz = fbm(p + vec3(91.7, 45.3, 67.1), g.noiseWarpFreq, detailOct, PERSISTENCE_DETAIL, float(g.noiseSeed) + 521.0);
    vec3 warp = vec3(fx, fy, fz);
    return p + warp * g.noiseWarpAmp;
}

float F_crust(in vec3 p) {
    float r = length(p);
    if (r <= 0.0) return -g.noiseMinHeight;
    vec3 dir = p / r;
    vec3 warped = domainWarp(dir);
    int contOct = max(int(g.noiseContinentOctaves), 1);
    int detailOct = max(int(g.noiseDetailOctaves), 1);
    int caveOct = max(int(g.noiseCaveOctaves), 1);
    float continents = fbm(warped, g.noiseContinentFreq, contOct, PERSISTENCE_CONTINENT, float(g.noiseSeed));
    float detail = fbm(warped * 2.0, g.noiseDetailFreq, detailOct, PERSISTENCE_DETAIL, float(g.noiseSeed) + 613.0);
    float height = g.noiseContinentAmp * continents + g.noiseDetailAmp * detail;
    height = clamp(height, g.noiseMinHeight, g.noiseMaxHeight);
    float surfaceRadius = g.planetRadius + height;
    float field = r - surfaceRadius;
    if (field < 0.0 && g.noiseCaveAmp > 0.0) {
        float cave = fbm(p, g.noiseCaveFreq, caveOct, PERSISTENCE_CAVE, float(g.noiseSeed) + 997.0);
        float cavity = cave - g.noiseCaveThreshold;
        if (cavity > 0.0) {
            field += -g.noiseCaveAmp * cavity;
        }
    }
    return field;
}

vec3 gradF(in vec3 p) {
    const float eps = g.voxelSize * 0.5;
    float fx1 = F_crust(p + vec3(eps, 0.0, 0.0));
    float fx0 = F_crust(p - vec3(eps, 0.0, 0.0));
    float fy1 = F_crust(p + vec3(0.0, eps, 0.0));
    float fy0 = F_crust(p - vec3(0.0, eps, 0.0));
    float fz1 = F_crust(p + vec3(0.0, 0.0, eps));
    float fz0 = F_crust(p - vec3(0.0, 0.0, eps));
    vec3 n = vec3(fx1 - fx0, fy1 - fy0, fz1 - fz0);
    float len = length(n);
    return (len > 0.0) ? (n / len) : vec3(0.0, 1.0, 0.0);
}

bool intersectSphere(in vec3 o, in vec3 d, in float R, out float t0, out float t1) {
    float a = dot(d, d);
    if (a <= 0.0) return false;
    float b = dot(o, d);
    float c = dot(o, o) - R * R;
    float disc = b * b - a * c;
    if (disc < 0.0) return false;
    float s = sqrt(max(disc, 0.0));
    float invA = 1.0 / a;
    float tnear = (-b - s) * invA;
    float tfar  = (-b + s) * invA;
    if (tnear > tfar) { float tmp = tnear; tnear = tfar; tfar = tmp; }
    t0 = tnear; t1 = tfar;
    return true;
}

// Intersect a ray with spherical shell Rin..Rout. Returns nearest forward interval.
bool intersectSphereShell(in vec3 o, in vec3 d, in float Rin, in float Rout, out float tEnter, out float tExit) {
    if (!(Rout > Rin && Rin >= 0.0)) return false;
    float to0, to1; // outer
    if (!intersectSphere(o, d, Rout, to0, to1)) return false;
    const float eps = 1e-6;
    float outerStart = max(to0, eps);
    float outerEnd   = to1;
    if (outerEnd <= outerStart) return false;

    float ti0, ti1;
    bool hitInner = intersectSphere(o, d, Rin, ti0, ti1);
    if (!hitInner) { tEnter = outerStart; tExit = outerEnd; return true; }

    float c1Start = outerStart;
    float c1End   = min(outerEnd, ti0);
    if (c1End > c1Start) { tEnter = c1Start; tExit = c1End; return true; }
    float c2Start = max(outerStart, ti1);
    float c2End   = outerEnd;
    if (c2End > c2Start) { tEnter = c2Start; tExit = c2End; return true; }
    return false;
}
