#include "app/App.h"
#include "core/Logging.h"
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    // Initialize logging early so platform callbacks (GLFW/Vulkan) use it
    auto cfg = core::determine_log_config(argc, argv, spdlog::level::info);
    core::init_logging(cfg);
    app::App app;
    return app.run();
}
