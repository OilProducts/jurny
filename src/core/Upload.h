#pragma once

// Upload — staging rings and timeline semaphores for batched uploads.
namespace core {
class UploadContext {
public:
    void init();
    void flush();
    void shutdown();
};
}

