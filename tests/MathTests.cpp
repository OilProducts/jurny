// Minimal assertions for spherical math M0.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/vec3.hpp>

#include "math/Spherical.h"

using math::CrustSample;
using math::F_crust;
using math::gradF;
using math::IntersectSphereShell;
using math::NoiseParams;
using math::PlanetParams;
using math::SampleCrust;

static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    PlanetParams P{ /*R*/10'000.0, /*T*/300.0, /*sea*/10'000.0, /*Hmax*/2'000.0 };
    NoiseParams disabled = NoiseParams::disabled();

    // F_crust on the sphere
    {
        // On the surface: r = R ⇒ F ≈ 0
        float f0 = F_crust({static_cast<float>(P.R), 0.0f, 0.0f}, P);
        assert(std::fabs(f0) < 1e-3f);
        // Outside vs inside
        float fOut = F_crust({static_cast<float>(P.R + 5.0), 0.0f, 0.0f}, P);
        float fIn  = F_crust({static_cast<float>(P.R - 5.0), 0.0f, 0.0f}, P);
        assert(fOut > 0.0f && fIn < 0.0f);
        // Gradient points radially
        auto g = gradF({1.0f, 2.0f, 3.0f}, P, 0.5f);
        // g should be unit length and parallel to p
        float len = std::sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
        assert(std::fabs(len - 1.0f) < 1e-3f);
    }

    // SampleCrust with disabled noise should match the analytic sphere
    {
        glm::vec3 p{static_cast<float>(P.R) + 25.0f, 12.0f, -3.0f};
        CrustSample s = SampleCrust(p, P, disabled, /*seed*/1337u);
        float analytic = F_crust(p, P);
        assert(approx(s.field, analytic, 1e-5f));
        assert(approx(s.height, 0.0f, 1e-5f));
    }

    // Shell intersection basics (use small radii for simple numbers)
    {
        const float R  = 12.0f;
        const float Rin= 9.0f;
        const float Rout=12.0f;

        // Outside, pointing inward along -X
        float t0=0, t1=0;
        bool hit = IntersectSphereShell({15.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        assert(hit);
        // Expect [3, 6] (enter outer at 15-12=3; enter inner at 15-9=6)
        assert(approx(t0, 3.0f, 1e-4f));
        assert(approx(t1, 6.0f, 1e-4f));

        // Inside shell, pointing +X: expect [~0, 1.5]
        hit = IntersectSphereShell({10.5f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        assert(hit);
        assert(t0 >= 0.0f && approx(t1, 1.5f, 1e-4f));

        // Inside inner sphere, pointing +X: expect [4, 7]
        hit = IntersectSphereShell({5.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        assert(hit);
        assert(approx(t0, 4.0f, 1e-4f) && approx(t1, 7.0f, 1e-4f));

        // Miss: above Rout, direction parallel to shell
        hit = IntersectSphereShell({0.0f, 0.0f, 20.0f}, {1.0f, 0.0f, 0.0f}, Rin, Rout, t0, t1);
        assert(!hit);
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
        assert(std::fabs(a.height - b.height) < 0.1f);
    }

    return 0;
}
