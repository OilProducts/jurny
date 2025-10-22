#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace fs = std::filesystem;

struct Options {
    fs::path dataDir;
    fs::path outDir;
    fs::path pakFile;
    fs::path manifestFile;
    fs::path indexFile;
    bool verbose = false;
};

struct AssetEntry {
    fs::path source;
    std::string logicalPath;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint64_t hash = 0;
};

[[nodiscard]] std::string escape_json(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += ch; break;
        }
    }
    return out;
}

[[nodiscard]] std::string hex64(std::uint64_t value) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << value;
    return ss.str();
}

[[nodiscard]] Options parse_cli(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg == "--data-dir" && i + 1 < argc) {
            opts.dataDir = fs::path(argv[++i]);
        } else if (arg == "--out-dir" && i + 1 < argc) {
            opts.outDir = fs::path(argv[++i]);
        } else if (arg == "--pak" && i + 1 < argc) {
            opts.pakFile = fs::path(argv[++i]);
        } else if (arg == "--manifest" && i + 1 < argc) {
            opts.manifestFile = fs::path(argv[++i]);
        } else if (arg == "--index" && i + 1 < argc) {
            opts.indexFile = fs::path(argv[++i]);
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: pack_assets --data-dir DIR --out-dir DIR [--pak FILE] "
                         "[--manifest FILE] [--index FILE] [--verbose]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    if (opts.dataDir.empty()) {
        std::cerr << "--data-dir is required.\n";
        std::exit(1);
    }
    if (opts.outDir.empty()) {
        std::cerr << "--out-dir is required.\n";
        std::exit(1);
    }
    if (opts.pakFile.empty()) {
        opts.pakFile = opts.outDir / "assets.pak";
    }
    if (opts.manifestFile.empty()) {
        opts.manifestFile = opts.outDir / "assets_manifest.json";
    }
    if (opts.indexFile.empty()) {
        opts.indexFile = opts.outDir / "assets_index.txt";
    }
    return opts;
}

std::vector<AssetEntry> gather_assets(const fs::path& root) {
    std::vector<AssetEntry> assets;
    if (!fs::exists(root)) {
        return assets;
    }

    for (auto it = fs::recursive_directory_iterator(root); it != fs::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file()) {
            continue;
        }
        const auto rel = fs::relative(it->path(), root);
        std::string logical = rel.generic_string();
        if (logical.empty()) {
            continue;
        }
        AssetEntry entry;
        entry.source = it->path();
        entry.logicalPath = logical;
        assets.push_back(std::move(entry));
    }

    std::sort(assets.begin(), assets.end(), [](const AssetEntry& a, const AssetEntry& b) {
        return a.logicalPath < b.logicalPath;
    });
    return assets;
}

void write_pak(const fs::path& pakPath, std::vector<AssetEntry>& assets) {
    fs::create_directories(pakPath.parent_path());
    std::ofstream out(pakPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open pak file for writing: " + pakPath.string());
    }

    std::uint64_t offset = 0;
    for (auto& asset : assets) {
        std::ifstream in(asset.source, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to read asset: " + asset.source.string());
        }

        std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        asset.size = static_cast<std::uint64_t>(buffer.size());
        asset.offset = offset;
        asset.hash = XXH3_64bits(buffer.data(), buffer.size());

        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (!out) {
            throw std::runtime_error("Failed to write asset data to pak.");
        }
        offset += asset.size;
    }
}

void write_manifest(const fs::path& manifestPath, const std::vector<AssetEntry>& assets) {
    fs::create_directories(manifestPath.parent_path());
    std::ofstream out(manifestPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open manifest for writing: " + manifestPath.string());
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &now_time);
#else
    gmtime_r(&now_time, &tm);
#endif
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm);

    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"generated_at\": \"" << timestamp << "\",\n";
    out << "  \"assets\": [\n";
    for (std::size_t i = 0; i < assets.size(); ++i) {
        const auto& asset = assets[i];
        out << "    {\n";
        out << "      \"path\": \"" << escape_json(asset.logicalPath) << "\",\n";
        out << "      \"offset\": " << asset.offset << ",\n";
        out << "      \"size\": " << asset.size << ",\n";
        out << "      \"hash_xxhash64\": \"0x" << hex64(asset.hash) << "\"\n";
        out << "    }";
        out << (i + 1 == assets.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void write_index(const fs::path& indexPath, const std::vector<AssetEntry>& assets) {
    fs::create_directories(indexPath.parent_path());
    std::ofstream out(indexPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open index for writing: " + indexPath.string());
    }
    out << "# path\toffset\tsize\thash_xxhash64\n";
    for (const auto& asset : assets) {
        out << asset.logicalPath << '\t'
            << asset.offset << '\t'
            << asset.size << '\t'
            << "0x" << hex64(asset.hash) << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        Options opts = parse_cli(argc, argv);
        opts.dataDir = fs::absolute(opts.dataDir);
        opts.outDir = fs::absolute(opts.outDir);
        opts.pakFile = fs::absolute(opts.pakFile);
        opts.manifestFile = fs::absolute(opts.manifestFile);
        opts.indexFile = fs::absolute(opts.indexFile);

        auto assets = gather_assets(opts.dataDir);
        if (assets.empty()) {
            fs::create_directories(opts.outDir);
            std::ofstream(opts.pakFile, std::ios::binary | std::ios::trunc).close();
            write_manifest(opts.manifestFile, assets);
            write_index(opts.indexFile, assets);
            if (opts.verbose) {
                std::cout << "No assets found; wrote empty pack.\n";
            }
            return 0;
        }

        write_pak(opts.pakFile, assets);
        write_manifest(opts.manifestFile, assets);
        write_index(opts.indexFile, assets);

        if (opts.verbose) {
            std::cout << "Packed " << assets.size() << " assets into " << opts.pakFile << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "pack_assets error: " << e.what() << "\n";
        return 1;
    }
}
