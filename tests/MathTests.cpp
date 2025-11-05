// Minimal assertions for spherical math M0.
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "math/Spherical.h"

using math::CrustSample;
using math::F_crust;
using math::gradF;
using math::IntersectSphereShell;
using math::NoiseParams;
using math::PlanetParams;
using math::SampleCrust;

static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

namespace {

void logFailureFmt(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "MathTests failure (%s:%d): ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

} // namespace

#define CHECK(cond, msg, ...) \
    do { \
        if (!(cond)) { \
            logFailureFmt(__FILE__, __LINE__, msg, ##__VA_ARGS__); \
            success = false; \
        } \
    } while (0)

int main() {
    PlanetParams P{ /*R*/10'000.0, /*T*/300.0, /*sea*/10'000.0, /*Hmax*/2'000.0 };
    NoiseParams disabled = NoiseParams::disabled();
    bool success = true;

    // F_crust sign changes around the actual surface radius
    {
        const glm::vec3 axis(1.0f, 0.0f, 0.0f);
        const float baseRadius = static_cast<float>(P.R);
        CrustSample surfaceSample = SampleCrust(axis * baseRadius, P, disabled, 0u);
        const float surfaceRadius = baseRadius + surfaceSample.height;
        CHECK(std::fabs(surfaceSample.field + surfaceSample.height) < 1e-3f,
              "Field/height relation should hold on surface (field=%g height=%g)",
              surfaceSample.field, surfaceSample.height);

        const float outsideRadius = surfaceRadius + 5.0f;
        const float insideRadius  = std::max(surfaceRadius - 5.0f, 0.0f);
        float fOut = F_crust(axis * outsideRadius, P);
        float fIn  = F_crust(axis * insideRadius, P);
        CHECK(fOut > 0.0f, "Outside surface should return positive (fOut=%g)", fOut);
        CHECK(fIn < 0.0f, "Inside surface should return negative (fIn=%g)", fIn);

        auto g = gradF({1.0f, 2.0f, 3.0f}, P, 0.5f);
        float len = std::sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
        CHECK(std::fabs(len - 1.0f) < 1e-3f, "Gradient magnitude should be ~1 (len=%g)", len);
    }

    // SampleCrust with disabled noise should match the analytic sphere
    {
        glm::vec3 p{static_cast<float>(P.R) + 25.0f, 12.0f, -3.0f};
        CrustSample s = SampleCrust(p, P, disabled, /*seed*/1337u);
        float analytic = F_crust(p, P);
        CHECK(approx(s.field, analytic, 1e-5f), "Field should match analytic sphere with noise disabled (s.field=%g analytic=%g)", s.field, analytic);
        const float expectedHeight = glm::length(p) - static_cast<float>(P.R) - s.field;
        CHECK(approx(s.height, expectedHeight, 1e-3f),
              "Height should satisfy r - R - field (height=%g expected=%g)", s.height, expectedHeight);
    }

    // Shell intersection basics (use small radii for simple numbers)
    {
        const float Rin= 9.0f;
        const float Rout=12.0f;

        // Outside, pointing inward along -X
        float t0=0, t1=0;
        bool hit = IntersectSphereShell({15.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        CHECK(hit, "Ray from outside should hit shell");
        // Expect [3, 6] (enter outer at 15-12=3; enter inner at 15-9=6)
        CHECK(approx(t0, 3.0f, 1e-4f), "Shell entry should be at 3 (t0=%g)", t0);
        CHECK(approx(t1, 6.0f, 1e-4f), "Shell exit should be at 6 (t1=%g)", t1);

        // Inside shell, pointing +X: expect [~0, 1.5]
        hit = IntersectSphereShell({10.5f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        CHECK(hit, "Ray inside shell should hit");
        CHECK(t0 >= 0.0f, "Ray inside shell should have non-negative entry (t0=%g)", t0);
        CHECK(approx(t1, 1.5f, 1e-4f), "Ray inside shell should exit at 1.5 (t1=%g)", t1);

        // Inside inner sphere, pointing +X: expect [4, 7]
        hit = IntersectSphereShell({5.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        CHECK(hit, "Ray from inside inner sphere should hit shell");
        CHECK(approx(t0, 4.0f, 1e-4f), "Inner entry should be 4 (t0=%g)", t0);
        CHECK(approx(t1, 7.0f, 1e-4f), "Outer exit should be 7 (t1=%g)", t1);

        // Miss: above Rout, direction parallel to shell
        hit = IntersectSphereShell({0.0f, 0.0f, 20.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        CHECK(!hit, "Ray parallel to shell should miss");
    }

    // Noise continuity: tiny perturbations should not introduce discontinuities.
    {
        PlanetParams Pw{100.0, 12.0, 100.0, 24.0};
        NoiseParams noise{};
        const std::uint32_t seed = 1337u;
        const float radius = static_cast<float>(Pw.R) + 5.0f;
        const glm::vec3 basePos(radius, 0.0f, 0.0f);
        const float eps = 1e-3f;
        CrustSample a = SampleCrust(basePos - glm::vec3(eps, 0.0f, 0.0f), Pw, noise, seed);
        CrustSample b = SampleCrust(basePos + glm::vec3(eps, 0.0f, 0.0f), Pw, noise, seed);
        CHECK(std::fabs(a.height - b.height) < 0.1f, "Noise continuity should be smooth (|a.height - b.height|=%g)", std::fabs(a.height - b.height));
    }

    return success ? 0 : 1;
}
