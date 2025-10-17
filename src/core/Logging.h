#pragma once

#include <string>
#include <string_view>
#include <spdlog/spdlog.h>

namespace core {

// Parse a log level string like "trace|debug|info|warn|error|critical|off".
// Returns fallback when unknown/empty.
spdlog::level::level_enum parse_log_level(std::string_view s,
                                          spdlog::level::level_enum fallback = spdlog::level::info);

struct LogInitConfig {
    spdlog::level::level_enum level = spdlog::level::info;
    std::string file_path; // empty => no file sink
    bool use_color = true;
};

// Inspect env and CLI to pick a log level and optional file sink.
// Supported CLI flags (order-agnostic):
//   --log-level <level>
//   --log-file <path>
//   --no-color
// Env vars:
//   VOXEL_LOG_LEVEL, VOXEL_LOG_FILE
LogInitConfig determine_log_config(int argc, char** argv,
                                   spdlog::level::level_enum default_level = spdlog::level::info);

// Initialize spdlog with a color console sink and optional rotating file sink.
// Applies a consistent pattern and periodic flushing.
void init_logging(const LogInitConfig& cfg);

// Optional shutdown helper (flushes and releases sinks).
void shutdown_logging();

} // namespace core

