#pragma once

// Raytracer — wavefront compute passes (generate → traverse → shade → denoise → composite).
namespace render {
class Raytracer {
public:
    void init();
    void render();
    void shutdown();
};
}

