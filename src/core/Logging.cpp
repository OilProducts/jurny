#include "Logging.h"

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace core {

static std::string_view trim(std::string_view s) {
    auto is_space = [](unsigned char c){ return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    size_t b = 0, e = s.size();
    while (b < e && is_space(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && is_space(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e - b);
}

spdlog::level::level_enum parse_log_level(std::string_view s, spdlog::level::level_enum fallback) {
    s = trim(s);
    if (s.empty()) return fallback;
    // Accept common aliases
    if (s == "trace") return spdlog::level::trace;
    if (s == "debug") return spdlog::level::debug;
    if (s == "info") return spdlog::level::info;
    if (s == "warn" || s == "warning") return spdlog::level::warn;
    if (s == "error" || s == "err") return spdlog::level::err;
    if (s == "critical" || s == "crit" || s == "fatal") return spdlog::level::critical;
    if (s == "off" || s == "none" || s == "quiet") return spdlog::level::off;
    return fallback;
}

LogInitConfig determine_log_config(int argc, char** argv, spdlog::level::level_enum default_level) {
    LogInitConfig cfg{}; cfg.level = default_level; cfg.use_color = true;

    // Env vars first
    if (const char* env_lvl = std::getenv("VOXEL_LOG_LEVEL")) {
        cfg.level = parse_log_level(env_lvl, cfg.level);
    }
    if (const char* env_file = std::getenv("VOXEL_LOG_FILE")) {
        if (env_file[0] != '\0') cfg.file_path = env_file;
    }

    // CLI overrides env
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i] ? argv[i] : "";
        if (a == "--log-level" && i + 1 < argc) {
            cfg.level = parse_log_level(argv[i + 1], cfg.level);
            ++i;
        } else if (a.rfind("--log-level=", 0) == 0) {
            cfg.level = parse_log_level(a.substr(std::string("--log-level=").size()), cfg.level);
        } else if (a == "--log-file" && i + 1 < argc) {
            cfg.file_path = argv[++i];
        } else if (a.rfind("--log-file=", 0) == 0) {
            cfg.file_path = std::string(a.substr(std::string("--log-file=").size()));
        } else if (a == "--no-color") {
            cfg.use_color = false;
        }
    }
    return cfg;
}

void init_logging(const LogInitConfig& cfg) {
    // Pattern with time, thread id, level (colored), and message
    const char* pattern = "[%H:%M:%S.%e] [tid %t] [%^%l%$] %v";

    std::vector<spdlog::sink_ptr> sinks;
    if (cfg.use_color) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    } else {
        // Fallback to non-color stdout if disabled
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
    }

    if (!cfg.file_path.empty()) {
        // 5 MB per file, keep 3 files
        constexpr std::size_t max_size = 5 * 1024 * 1024;
        constexpr std::size_t max_files = 3;
        try {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(cfg.file_path, max_size, max_files);
            sinks.push_back(file_sink);
        } catch (...) {
            // If file sink fails (permissions, path), keep console only
        }
    }

    auto logger = std::make_shared<spdlog::logger>("voxel", sinks.begin(), sinks.end());
    logger->set_level(cfg.level);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);
    spdlog::set_level(cfg.level);
    spdlog::set_pattern(pattern);

    // Periodic flush to keep logs fresh without flushing on every info
    spdlog::flush_every(std::chrono::seconds(1));

    spdlog::info("Logging initialized (level={}, file={})",
                 spdlog::level::to_string_view(cfg.level),
                 cfg.file_path.empty() ? "none" : cfg.file_path.c_str());
}

void shutdown_logging() {
    spdlog::shutdown();
}

} // namespace core
