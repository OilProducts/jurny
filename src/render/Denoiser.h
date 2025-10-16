#pragma once

// Denoiser — temporal reprojection + A-trous filtering.
namespace render {
class Denoiser {
public:
    void init();
    void execute();
    void shutdown();
};
}

