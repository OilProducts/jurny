// spherical.glsl — planet signed field F(p), gradF, and shell/sphere intersection.
// Keep in sync with world/WorldGen.cpp for crustField.

uint pcgHash32(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(uint seed) {
    seed = pcgHash32(seed);
    return float(seed) * (1.0 / 4294967296.0);
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

float valueNoise(vec3 p, uint seed) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = fade3(f);
    ivec3 cell = ivec3(i);
    uint baseSeed = seed;
    float c000 = randomFloat(latticeHash(cell + ivec3(0, 0, 0), baseSeed));
    float c100 = randomFloat(latticeHash(cell + ivec3(1, 0, 0), baseSeed));
    float c010 = randomFloat(latticeHash(cell + ivec3(0, 1, 0), baseSeed));
    float c110 = randomFloat(latticeHash(cell + ivec3(1, 1, 0), baseSeed));
    float c001 = randomFloat(latticeHash(cell + ivec3(0, 0, 1), baseSeed));
    float c101 = randomFloat(latticeHash(cell + ivec3(1, 0, 1), baseSeed));
    float c011 = randomFloat(latticeHash(cell + ivec3(0, 1, 1), baseSeed));
    float c111 = randomFloat(latticeHash(cell + ivec3(1, 1, 1), baseSeed));

    float nx00 = mix(c000, c100, u.x);
    float nx10 = mix(c010, c110, u.x);
    float nx01 = mix(c001, c101, u.x);
    float nx11 = mix(c011, c111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

float signedValueNoise(vec3 p, uint seed) {
    return valueNoise(p, seed) * 2.0 - 1.0;
}

float ridgeSignal(float v, float sharpness) {
    float ridge = 1.0 - abs(v);
    ridge = clamp(ridge, 0.0, 1.0);
    return (sharpness > 0.0) ? (pow(ridge, sharpness) * 2.0 - 1.0) : (ridge * 2.0 - 1.0);
}

float macroLayer(vec3 surfacePoint, vec3 dir) {
    if (g.noiseMacroAmp <= 0.0 || g.noiseMacroFreq <= 0.0) return 0.0;
    vec3 domain = surfacePoint * g.noiseMacroFreq;
    float base = signedValueNoise(domain, g.noiseSeed);
    float ridgeSrc = signedValueNoise(domain + vec3(19.1, 7.7, 13.3), g.noiseSeed + 17u);
    float ridge = ridgeSignal(ridgeSrc, max(g.noiseMacroSharpness, 0.5));
    float jitter = signedValueNoise(domain * 0.5 + vec3(31.7, 23.1, 11.9), g.noiseSeed + 31u);
    float macro = mix(base, ridge, clamp(g.noiseMacroRidgeWeight, 0.0, 1.0));
    macro = mix(macro, jitter, 0.15) + abs(dir.z) * 0.05;
    return g.noiseMacroAmp * macro;
}

float detailLayer(vec3 surfacePoint) {
    if (g.noiseDetailAmp <= 0.0 || g.noiseDetailFreq <= 0.0) return 0.0;
    vec3 domain = surfacePoint * g.noiseDetailFreq;
    float base = signedValueNoise(domain + vec3(3.1, 7.3, 11.9), g.noiseSeed + 101u);
    float ridgeSrc = signedValueNoise(domain * 1.7 + vec3(17.0, 5.0, 9.0), g.noiseSeed + 131u);
    float ridge = ridgeSignal(ridgeSrc, max(g.noiseDetailSharpness, 0.5));
    float trig = sin(dot(domain, vec3(0.8, 1.1, 0.5)) + 1.3) *
                 cos(dot(domain, vec3(1.5, 0.4, 1.7)) - 0.6);
    trig = clamp(trig, -1.0, 1.0);
    float detail = mix(base, ridge, clamp(g.noiseDetailRidgeWeight, 0.0, 1.0));
    detail = mix(detail, trig, 0.25);
    return g.noiseDetailAmp * detail;
}

float latitudeBands(vec3 dir) {
    if (g.noiseBandAmp <= 0.0 || g.noiseBandFreq <= 0.0) return 0.0;
    float lat = dir.z * 0.5 + 0.5;
    float s = sin(lat * g.noiseBandFreq * (2.0 * 3.14159265));
    float falloff = exp(-g.noiseBandSharpness * (1.0 - abs(dir.z)));
    return g.noiseBandAmp * s * falloff;
}

float cavesContribution(vec3 p) {
    if (g.noiseCavityAmp <= 0.0 || g.noiseCavityFreq <= 0.0) return 0.0;
    vec3 domain = p * g.noiseCavityFreq;
    float base = signedValueNoise(domain, g.noiseSeed + 997u);
    float pocket = signedValueNoise(domain + vec3(13.0, 29.0, 17.0), g.noiseSeed + 1019u);
    float signal = mix(base, pocket, 0.5) * 0.5 + 0.5;
    float contrast = max(g.noiseCavityContrast, 0.01);
    float mask = clamp((signal - clamp(g.noiseCavityThreshold, 0.0, 1.0)) * contrast, 0.0, 1.0);
    mask *= mask;
    return -g.noiseCavityAmp * mask;
}

float F_crust(in vec3 p) {
    float r = length(p);
    if (r <= 0.0) return -g.noiseMinHeight;
    vec3 dir = p / r;
    float baseRadius = g.planetRadius;
    vec3 surface = dir * baseRadius;

    float height = g.noiseBaseHeightOffset;
    height += macroLayer(surface, dir);
    height += detailLayer(surface);
    height += latitudeBands(dir);
    height = clamp(height, g.noiseMinHeight, g.noiseMaxHeight);

    float field = r - (baseRadius + height);
    if (field < 0.0) {
        field += cavesContribution(p);
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
