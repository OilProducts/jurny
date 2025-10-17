// spherical.glsl — planet signed field F(p), gradF, and shell/sphere intersection.
// Keep in sync with src/math/Spherical.cpp.

float F_crust(in vec3 p, in float R) {
    return length(p) - R;
}

vec3 gradF(in vec3 p) {
    float r = length(p);
    return (r > 0.0) ? (p / r) : vec3(0.0, 1.0, 0.0);
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
