#pragma once

// Pipelines — compute pipeline cache and hot-reload.
namespace core {
class Pipelines {
public:
    void init();
    void reloadIfChanged();
    void shutdown();
};
}

