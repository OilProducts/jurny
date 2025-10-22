#include "Assets.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>

#include <spdlog/spdlog.h>

namespace core {

namespace {
std::string normalizePath(std::string_view path) {
    std::string out(path);
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}

bool parseIndexLine(const std::string& line,
                    std::string& path,
                    AssetRegistry::Entry& entry) {
    if (line.empty() || line[0] == '#') {
        return false;
    }
    std::istringstream ss(line);
    std::string offsetStr;
    std::string sizeStr;
    std::string hashStr;
    if (!std::getline(ss, path, '\t')) return false;
    if (!std::getline(ss, offsetStr, '\t')) return false;
    if (!std::getline(ss, sizeStr, '\t')) return false;
    if (!std::getline(ss, hashStr, '\t')) return false;

    try {
        entry.offset = static_cast<std::uint64_t>(std::stoull(offsetStr));
        entry.size = static_cast<std::uint64_t>(std::stoull(sizeStr));
        if (!hashStr.empty() && hashStr.rfind("0x", 0) == 0) {
            entry.hash = std::stoull(hashStr, nullptr, 16);
        } else {
            entry.hash = static_cast<std::uint64_t>(std::stoull(hashStr));
        }
    } catch (...) {
        return false;
    }
    return true;
}
} // namespace

bool AssetRegistry::initialize(const std::string& assetDir) {
    namespace fs = std::filesystem;
    fs::path base(assetDir);
    fs::path pak = base / "assets.pak";
    fs::path index = base / "assets_index.txt";
    return initialize(pak.string(), index.string());
}

bool AssetRegistry::initialize(const std::string& pakPath, const std::string& indexPath) {
    entries_.clear();
    assetOrder_.clear();
    pakData_.clear();

    if (!readPak(pakPath)) {
        spdlog::error("AssetRegistry: failed to read pak '{}'", pakPath);
        return false;
    }
    if (!readIndex(indexPath)) {
        spdlog::error("AssetRegistry: failed to read index '{}'", indexPath);
        entries_.clear();
        assetOrder_.clear();
        pakData_.clear();
        return false;
    }
    return true;
}

bool AssetRegistry::contains(std::string_view path) const {
    return entries_.find(normalizePath(path)) != entries_.end();
}

bool AssetRegistry::getEntry(std::string_view path, Entry& out) const {
    auto it = entries_.find(normalizePath(path));
    if (it == entries_.end()) return false;
    out = it->second;
    if (out.offset + out.size > pakData_.size()) {
        spdlog::error("AssetRegistry: entry '{}' points outside pack (offset {} size {}, pack size {})",
                      path,
                      static_cast<unsigned long long>(out.offset),
                      static_cast<unsigned long long>(out.size),
                      static_cast<unsigned long long>(pakData_.size()));
        return false;
    }
    return true;
}

std::span<const std::byte> AssetRegistry::view(std::string_view path) const {
    Entry entry{};
    if (!getEntry(path, entry)) {
        return {};
    }
    const std::byte* begin = pakData_.data() + entry.offset;
    return { begin, static_cast<std::size_t>(entry.size) };
}

bool AssetRegistry::loadBinary(std::string_view path, std::vector<std::uint8_t>& out) const {
    Entry entry{};
    if (!getEntry(path, entry)) {
        return false;
    }
    out.resize(static_cast<std::size_t>(entry.size));
    const std::byte* src = pakData_.data() + entry.offset;
    std::memcpy(out.data(), src, out.size());
    return true;
}

bool AssetRegistry::loadText(std::string_view path, std::string& out) const {
    Entry entry{};
    if (!getEntry(path, entry)) {
        return false;
    }
    const std::byte* src = pakData_.data() + entry.offset;
    out.assign(reinterpret_cast<const char*>(src),
               reinterpret_cast<const char*>(src) + entry.size);
    return true;
}

bool AssetRegistry::readPak(const std::string& pakPath) {
    std::ifstream in(pakPath, std::ios::binary | std::ios::ate);
    if (!in) {
        spdlog::error("AssetRegistry: unable to open '{}'", pakPath);
        return false;
    }
    const auto size = in.tellg();
    if (size <= 0) {
        pakData_.clear();
        return true;
    }
    pakData_.resize(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(pakData_.data()), size);
    if (!in) {
        spdlog::error("AssetRegistry: failed to read '{}' ({} bytes requested)", pakPath,
                      static_cast<long long>(size));
        pakData_.clear();
        return false;
    }
    return true;
}

bool AssetRegistry::readIndex(const std::string& indexPath) {
    std::ifstream in(indexPath);
    if (!in) {
        spdlog::error("AssetRegistry: unable to open '{}'", indexPath);
        return false;
    }
    std::string line;
    std::string path;
    Entry entry{};
    while (std::getline(in, line)) {
        if (!parseIndexLine(line, path, entry)) {
            continue;
        }
        std::string key = normalizePath(path);
        entries_[key] = entry;
        assetOrder_.push_back(key);
    }
    bool ok = true;
    for (const auto& [name, e] : entries_) {
        if (e.offset + e.size > pakData_.size()) {
            spdlog::error("AssetRegistry: '{}' entry exceeds pack size (offset {} size {}, pack {})",
                          name,
                          static_cast<unsigned long long>(e.offset),
                          static_cast<unsigned long long>(e.size),
                          static_cast<unsigned long long>(pakData_.size()));
            ok = false;
        }
    }
    return ok;
}

} // namespace core
